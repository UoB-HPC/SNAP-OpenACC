#include <stdlib.h>

#include "ext_sweep.h"
#include "ext_macros.h"
#include "ext_kernels.h"
#include "ext_problem.h"
#include "ext_profiler.h"

// Compute the order of the sweep for the first octant
plane *compute_sweep_order(void)
{
    START_PROFILING;

    unsigned int nplanes = ichunk + ny + nz - 2;
    plane *planes = (plane *)malloc(sizeof(plane)*nplanes);
    for (unsigned int i = 0; i < nplanes; i++)
    {
        planes[i].num_cells = 0;
    }

    // Cells on each plane have equal co-ordinate sum
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                planes[n].num_cells++;
            }
        }
    }

    // Allocate the memory for each plane
    for (unsigned int i = 0; i < nplanes; i++)
    {
        planes[i].cells = (struct cell *)malloc(sizeof(struct cell)*planes[i].num_cells);
        planes[i].index = 0;
    }

    // Store the cell indexes in the plane array
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                unsigned int idx = planes[n].index;
                planes[n].cells[idx].i = i;
                planes[n].cells[idx].j = j;
                planes[n].cells[idx].k = k;
                planes[n].index += 1;
            }
        }
    }

    STOP_PROFILING(__func__);

    return planes;
}

// Sweep over the grid and compute the angular flux
void sweep_octant(
        const unsigned int timestep, 
        const unsigned int oct, 
        const unsigned int ndiag, 
        const plane *planes, 
        const unsigned int num_groups_todo)
{
    // Determine the cell step parameters for the given octant
    // Create the list of octant co-ordinates in order

    // This first bit string assumes 3 reflective boundaries
    //int order_3d = 0b000001010100110101011111;

    // This bit string is lexiographically organised
    // This is the order to match the original SNAP
    // However this required all vacuum boundaries
    //int order_3d = 0b000001010011100101110111;
    int order_3d = 342391;

    //int order_2d = 0b11100100;
    int order_2d = 228;

    // Use the bit mask to get the right values for starting positions of the sweep
    int xhi = ((order_3d >> (oct * 3)) & 1) ? nx : 0;
    int yhi = ((order_3d >> (oct * 3 + 1)) & 1) ? ny : 0;
    int zhi = ((order_3d >> (oct * 3 + 2)) & 1) ? nz : 0;

    // Set the order you traverse each axis
    int istep = (xhi == nx) ? -1 : 1;
    int jstep = (yhi == ny) ? -1 : 1;
    int kstep = (zhi == nz) ? -1 : 1;

    size_t offset = oct*nang*nx*ny*nz*ng;
    double* l_flux_in = (timestep % 2 == 0 ? flux_in : flux_out) + offset;
    double* l_flux_out = (timestep % 2 == 0 ? flux_out : flux_in) + offset;

    for (unsigned int d = 0; d < ndiag; d++)
    {
        sweep_cell(istep, jstep, kstep, oct, l_flux_in, l_flux_out,
                planes[d].cells, groups_todo, num_groups_todo, planes[d].num_cells);
    }
}

// Perform a sweep over the grid for all the octants
void perform_sweep(
        unsigned int num_groups_todo)
{
    // Number of planes in this octant
    unsigned int ndiag = ichunk + ny + nz - 2;

    // Get the order of cells to enqueue
    plane *planes = compute_sweep_order();

    for (int o = 0; o < noct; o++)
    {
        sweep_octant(global_timestep, o, ndiag, planes, num_groups_todo);
        zero_edge_flux_buffers();
    }

    // Free planes
    for (unsigned int i = 0; i < ndiag; i++)
    {
        free(planes[i].cells);
    }

    free(planes);
}

// Solve the transport equations for a single angle in a single cell for a single group
void sweep_cell(
        const int istep,
        const int jstep,
        const int kstep,
        const unsigned int oct,
        const double* restrict l_flux_in,
        double* restrict l_flux_out,
        const struct cell * restrict cell_index,
        const unsigned int * restrict groups_todo,
        const unsigned int num_groups_todo,
        const unsigned int num_cells)
{
    START_PROFILING;

#pragma acc kernels \
    copyin(cell_index[:num_cells]), \
    present(groups_todo[:groups_todo_len], source[:source_len], \
            scat_coeff[:scat_coeff_len], flux_i[:flux_i_len], \
            flux_j[:flux_j_len], flux_k[:flux_k_len], dd_j[:dd_j_len], \
            dd_k[:dd_k_len], mu[:mu_len], l_flux_in[:l_flux_in_len], \
            l_flux_out[:l_flux_out_len], time_delta[:time_delta_len], \
            total_cross_section[:total_cross_section_len], denom[:denom_len])
#pragma acc loop independent collapse(3) vector(128)
    for(int nc = 0; nc < num_cells; ++nc)
    {
        for(int tg = 0; tg < num_groups_todo; ++tg)
        {
            for(int a = 0; a < nang; ++a)
            {
                // Get indexes for angle and group
                const unsigned int i = (istep > 0) ? cell_index[nc].i : nx - cell_index[nc].i - 1;
                const unsigned int j = (jstep > 0) ? cell_index[nc].j : ny - cell_index[nc].j - 1;
                const unsigned int k = (kstep > 0) ? cell_index[nc].k : nz - cell_index[nc].k - 1;
                const unsigned int g = groups_todo[tg];

                // Assume transmissive (vacuum boundaries) and that we
                // are sweeping the whole grid so have access to all neighbours
                // This means that we only consider the case for one MPI task
                // at present.

                // Compute angular source
                // Begin with first scattering moment
                double source_term = source(0,i,j,k,g);

                // Add in the anisotropic scattering source moments
                for (unsigned int l = 1; l < cmom; l++)
                {
                    source_term += scat_coeff(l,a,oct) * source(l,i,j,k,g);
                }

                double psi = source_term 
                    + flux_i(a,g,j,k)*mu(a)*dd_i 
                    + flux_j(a,g,i,k)*dd_j(a) 
                    + flux_k(a,g,i,j)*dd_k(a);

                // Add contribution from last timestep flux if time-dependant
                if (time_delta(g) != 0.0)
                {
                    psi += time_delta(g) * l_flux_in(a,g,i,j,k);
                }

                psi *= denom(a,g,i,j,k);

                // Compute upwind fluxes
                double tmp_flux_i = 2.0*psi - flux_i(a,g,j,k);
                double tmp_flux_j = 2.0*psi - flux_j(a,g,i,k);
                double tmp_flux_k = 2.0*psi - flux_k(a,g,i,j);

                // Time differencing on final flux value
                if (time_delta(g) != 0.0)
                {
                    psi = 2.0 * psi - l_flux_in(a,g,i,j,k);
                }

                // Perform the fixup loop
                int num_to_fix = 4;
                // Fixup is a bounded loop as we will worst case fix up each face and centre value one after each other
                double zeros[4];
//#pragma acc loop private(tmp_flux_i, tmp_flux_j, tmp_flux_k, psi)
                for (int fix = 0; fix < 4; fix++)
                {
                    // Record which ones are zero
                    zeros[0] = (tmp_flux_i < 0.0) ? 0.0 : 1.0;
                    zeros[1] = (tmp_flux_j < 0.0) ? 0.0 : 1.0;
                    zeros[2] = (tmp_flux_k < 0.0) ? 0.0 : 1.0;
                    zeros[3] = (psi < 0.0) ? 0.0 : 1.0;

                    if (num_to_fix == zeros[0] + zeros[1] + zeros[2] + zeros[3])
                    {
                        break;
                    }

                    num_to_fix = zeros[0] + zeros[1] + zeros[2] + zeros[3];

                    // Recompute cell centre value
                    psi = flux_i(a,g,j,k)*mu(a)*dd_i*(1.0+zeros[0]) 
                        + flux_j(a,g,j,k)*dd_j(a)*(1.0+zeros[1]) 
                        + flux_k(a,g,i,j)*dd_k(a)*(1.0+zeros[2]);

                    if (time_delta(g) != 0.0)
                    {
                        psi += time_delta(g) * l_flux_in(a,g,i,j,k) * (1.0+zeros[3]);
                    }
                    psi = 0.5*psi + source_term;
                    double recalc_denom = total_cross_section(g,i,j,k);
                    recalc_denom += mu(a) * dd_i * zeros[0];
                    recalc_denom += dd_j(a) * zeros[1];
                    recalc_denom += dd_k(a) * zeros[2];
                    recalc_denom += time_delta(g) * zeros[3];

                    if (recalc_denom > 1.0E-12)
                    {
                        psi /= recalc_denom;
                    }
                    else
                    {
                        psi = 0.0;
                    }

                    // Recompute the edge fluxes with the new centre value
                    tmp_flux_i = 2.0 * psi - flux_i(a,g,j,k);
                    tmp_flux_j = 2.0 * psi - flux_j(a,g,i,k);
                    tmp_flux_k = 2.0 * psi - flux_k(a,g,i,j);
                    if (time_delta(g) != 0.0)
                    {
                        psi = 2.0*psi - l_flux_in(a,g,i,j,k);
                    }
                }

                // Fix up loop is done, just need to set the final values
                tmp_flux_i = tmp_flux_i * zeros[0];
                tmp_flux_j = tmp_flux_j * zeros[1];
                tmp_flux_k = tmp_flux_k * zeros[2];
                psi = psi * zeros[3];

                // Write values to global memory
                flux_i(a,g,j,k) = tmp_flux_i;
                flux_j(a,g,i,k) = tmp_flux_j;
                flux_k(a,g,i,j) = tmp_flux_k;
                l_flux_out(a,g,i,j,k) = psi;
            }
        }
    }

    STOP_PROFILING(__func__);
}
