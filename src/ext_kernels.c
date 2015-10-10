#include <stdint.h>
#include <math.h>

#include "ext_sweep.h"
#include "ext_macros.h"
#include "ext_problem.h"
#include "ext_profiler.h"

// Calculate the inverted denominator for all the energy groups
void calc_denominator(void)
{
    START_PROFILING;

#pragma acc parallel loop \
    present(denom[:denom_len], total_cross_section[:total_cross_section_len], \
            time_delta[:time_delta_len], mu[:mu_len], dd_j[:dd_j_len], \
            dd_k[:dd_k_len])
    for (unsigned int ind = 0; ind < nx*ny*nz; ind++)
    {
        for (unsigned int g = 0; g < ng; ++g)
        {
            for (unsigned int a = 0; a < nang; ++a)
            {
                denom[a+g*nang+ind*ng*nang] = 1.0 / (total_cross_section[g+ind*ng] 
                        + time_delta(g) + mu(a)*dd_i + dd_j(a) + dd_k(a));
            }
        }
    }

    STOP_PROFILING(__func__, 1);
}

// Calculate the time delta
void calc_time_delta(void)
{
    START_PROFILING;

#pragma acc parallel loop \
    present(time_delta[:time_delta_len], velocity[:velocity_len])
    for(int g = 0; g < ng; ++g)
    {
        time_delta(g) = 2.0 / (dt * velocity(g));
    }

    STOP_PROFILING(__func__, 1);
}

// Calculate the diamond difference coefficients
void calc_dd_coefficients(void)
{
    START_PROFILING;

#pragma acc kernels \
    present(dd_j[:dd_j_len], dd_k[:dd_k_len], eta[:eta_len], xi[:xi_len])
    {
        dd_i = 2.0 / dx;

#pragma acc loop 
        for(int a = 0; a < nang; ++a)
        {
            dd_j(a) = (2.0/dy)*eta(a);
            dd_k(a) = (2.0/dz)*xi(a);
        }
    }

    STOP_PROFILING(__func__, 1);
}

// Calculate the total cross section from the spatial mapping
void calc_total_cross_section(void)
{
    START_PROFILING;

#pragma acc parallel loop \
    present(total_cross_section[:total_cross_section_len], xs[:xs_len], mat[:mat_len])
    for(int k = 0; k < nz; ++k)
    {
        for(int j = 0; j < ny; ++j)
        {
            for(int i = 0; i < nx; ++i)
            {
                for(int g = 0; g < ng; ++g)
                {
                    total_cross_section(g,i,j,k) = xs(mat(i,j,k)-1,g);
                }
            }
        }
    }

    STOP_PROFILING(__func__, 1);
}

void calc_scattering_cross_section(void)
{
    START_PROFILING;

#pragma acc parallel loop \
    present(scat_cs[:scat_cs_len], gg_cs[:gg_cs_len], mat[:mat_len])
    for(unsigned int g = 0; g < ng; ++g)
    {
        for (unsigned int k = 0; k < nz; k++)
        {
            for (unsigned int j = 0; j < ny; j++)
            {
                for (unsigned int i = 0; i < nx; i++)
                {
                    for (unsigned int l = 0; l < nmom; l++)
                    {
                        scat_cs(l,i,j,k,g) = gg_cs(mat(i,j,k)-1,l,g,g);
                    }
                }
            }
        }
    }

    STOP_PROFILING(__func__, 1);
}

// Calculate the outer source
void calc_outer_source(void)
{
    START_PROFILING;

#pragma acc parallel loop \
    present(g2g_source[:g2g_source_len], fixed_source[:fixed_source_len], \
            scalar_flux[:scalar_flux_len], mat[:mat_len], lma[:lma_len], \
            scalar_mom[:scalar_mom_len], gg_cs[:gg_cs_len])
    for (unsigned int g1 = 0; g1 < ng; g1++)
    {
        for(int k = 0; k < nz; ++k)
        {
            for(int j = 0; j < ny; ++j)
            {
                for(int i = 0; i < nx; ++i)
                {
                    g2g_source(0,i,j,k,g1) = fixed_source(i,j,k,g1);

                    for (unsigned int g2 = 0; g2 < ng; g2++)
                    {
                        if (g1 == g2)
                        {
                            continue;
                        }

                        g2g_source(0,i,j,k,g1) += gg_cs(mat(i,j,k)-1,0,g2,g1) * scalar_flux(g2,i,j,k);

                        unsigned int mom = 1;
                        for (unsigned int l = 1; l < nmom; l++)
                        {
                            for (int m = 0; m < lma(l); m++)
                            {
                                // TODO: CHECK WHY THIS CONDITION WAS NECESSARY
                                if(mom < (cmom-1))
                                {
                                    g2g_source(mom,i,j,k,g1) += gg_cs(mat(i,j,k)-1,l,g2,g1) 
                                        * scalar_mom(g2,mom-1,i,j,k);
                                }

                                mom++;
                            }
                        }
                    }
                }
            }
        }
    }

    STOP_PROFILING(__func__, 1);
}

// Calculate the inner source
void calc_inner_source(void)
{
    START_PROFILING;

#pragma acc parallel loop \
    present(source[:source_len], g2g_source[:g2g_source_len], scat_cs[:scat_cs_len], \
            scalar_flux[:scalar_flux_len], lma[:lma_len], scalar_mom[:scalar_mom_len])
    for (unsigned int g = 0; g < ng; g++)
    {
        for(int k = 0; k < nz; ++k)
        {
            for(int j = 0; j < ny; ++j)
            {
                for(int i = 0; i < nx; ++i)
                {
                    source(0,i,j,k,g) = g2g_source(0,i,j,k,g) + scat_cs(0,i,j,k,g) * scalar_flux(g,i,j,k);

                    unsigned int mom = 1;
                    for (unsigned int l = 1; l < nmom; l++)
                    {
                        for (int m = 0; m < lma(l); m++)
                        {
                            source(mom,i,j,k,g) = g2g_source(mom,i,j,k,g) + scat_cs(l,i,j,k,g) * scalar_mom(g,mom-1,i,j,k);
                            mom++;
                        }
                    }
                }
            }
        }
    }

    STOP_PROFILING(__func__, 1);
}

void zero_flux_in_out(void)
{
#pragma acc parallel loop \
    present(flux_in[:flux_in_len])
    for(int i = 0; i < flux_in_len; ++i)
    {
        flux_in[i] = 0.0;
    }

#pragma acc parallel loop \
    present(flux_out[:flux_out_len])
    for(int i = 0; i < flux_out_len; ++i)
    {
        flux_out[i] = 0.0;
    }
}

void zero_edge_flux_buffers(void)
{
    int fi_len = nang*ng*ny*nz;
    int fj_len = nang*ng*nx*nz;
    int fk_len = nang*ng*nx*ny;

#define MAX(A,B) (((A) > (B)) ? (A) : (B))
    int max_length = MAX(MAX(fi_len, fj_len), fk_len);

#pragma acc parallel loop \
    present(flux_i[:flux_i_len], flux_j[:flux_j_len], flux_k[:flux_k_len])
    for(int i = 0; i < max_length; ++i)
    {
        if(i < fi_len) flux_i[i] = 0.0;
        if(i < fj_len) flux_j[i] = 0.0;
        if(i < fk_len) flux_k[i] = 0.0;
    }
}

void zero_flux_moments_buffer(void)
{
#pragma acc parallel loop \
    present(scalar_mom[:scalar_mom_len])
    for(int i = 0; i < (cmom-1)*nx*ny*nz*ng; ++i)
    {
        scalar_mom[i] = 0.0;
    }
}

void zero_scalar_flux(void)
{
#pragma acc parallel loop \
    present(scalar_flux[:scalar_flux_len])
    for(int i = 0; i < nx*ny*nz*ng; ++i)
    {
        scalar_flux[i] = 0.0;
    }
}

int check_convergence(
        double *old, 
        double *new, 
        double epsi, 
        unsigned int *groups_todo, 
        unsigned int *num_groups_todo, 
        int inner)
{
    START_PROFILING;

    int r = 1;
    int ngt = 0;

#pragma acc parallel reduction(+:ngt) \
    present(old[:scalar_flux_len], new[:scalar_flux_len], groups_todo[:groups_todo_len])
    {
#pragma loop 
        for (unsigned int g = 0; g < ng; g++)
        {
            int gr = 0;
            for (unsigned int k = 0; k < nz; k++)
            {
                if (gr) break;
                for (unsigned int j = 0; j < ny; j++)
                {
                    if (gr) break;
                    for (unsigned int i = 0; i < nx; i++)
                    {
                        double val;
                        if (fabs(old[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)] > tolr))
                        {
                            val = fabs(new[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)]/old[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)] - 1.0);
                        }
                        else
                        {
                            val = fabs(new[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)] - old[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)]);
                        }

                        if (val > epsi)
                        {
                            if (inner)
                            {
                                gr = 1;
                            }

                            r = 0;
                            break;
                        }
                    }
                }
            }

            // Add g to the list of groups to do if we need to do it
            if (inner && gr)
            {
                groups_todo[ngt] = g;
                ngt += 1;
            }
        }

        // Check all inner groups are done in outer convergence test
        if (!inner)
        {
            if (ngt != 0)
            {
                r = 0;
            }
        }
    }

    if(inner)
    {
        *num_groups_todo = ngt;
    }
    else
    {
        *num_groups_todo += ngt;
    }

    STOP_PROFILING(__func__, 1);

    return r;
}

void initialise_device_memory(void)
{
   zero_scalar_flux();
   zero_edge_flux_buffers();
   zero_flux_moments_buffer();
   zero_flux_in_out();

#pragma acc parallel loop \
    present(g2g_source[:g2g_source_len])
    for(int ii = 0; ii < g2g_source_len; ++ii)
    {
        g2g_source[ii] = 0.0;
    }
}

// Copies the value of scalar flux
void store_scalar_flux(double* to)
{
    START_PROFILING;

#pragma acc parallel loop \
    present(scalar_flux[:scalar_flux_len], to[:scalar_flux_len])
    for(int i = 0; i < nx*ny*nz*ng; ++i)
    {
        to[i] = scalar_flux[i];
    }

    STOP_PROFILING(__func__, 1);
}
