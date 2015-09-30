#include <stdbool.h>
#include <math.h>

#include "ext_sweep.h"
#include "ext_macros.h"
#include "ext_problem.h"
#include "ext_profiler.h"

// Calculate the inverted denominator for all the energy groups
void calc_denominator(void)
{
    START_PROFILING;

#pragma acc parallel \
    present(denom[0:nang*ng*nx*ny*nz], total_cross_section[0:ng*nx*ny*nz], \
            time_delta[0:ng], mu[0:nang], dd_j[0:nang], dd_k[0:nang])
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < nx; i++)
            {
                for (unsigned int g = 0; g < ng; ++g)
                {
                    for (unsigned int a = 0; a < nang; ++a)
                    {
                        denom(a,g,i,j,k) = 1.0 / (total_cross_section(g,i,j,k)
                              + time_delta(g) + mu(a)*dd_i + dd_j(a) + dd_k(a));
                    }
                }
            }
        }
    }

    STOP_PROFILING(__func__, true);
}

// Calculate the time delta
void calc_time_delta(void)
{
    START_PROFILING;

#pragma acc parallel \
    present(time_delta[0:ng], velocity[0:ng])
    for(int g = 0; g < ng; ++g)
    {
        time_delta(g) = 2.0 / (dt * velocity(g));
    }

    STOP_PROFILING(__func__, true);
}

// Calculate the diamond difference coefficients
void calc_dd_coefficients(void)
{
    START_PROFILING;

    dd_i = 2.0 / dx;

#pragma acc parallel \
    present(dd_j[0:nang], dd_k[0:nang], eta[0:nang], xi[0:nang])
    for(int a = 0; a < nang; ++a)
    {
        dd_j(a) = (2.0/dy)*eta(a);
        dd_k(a) = (2.0/dz)*xi(a);
    }

    STOP_PROFILING(__func__, true);
}

// Calculate the total cross section from the spatial mapping
void calc_total_cross_section(void)
{
    START_PROFILING;

#pragma acc parallel \
    present(total_cross_section[0:ng*nx*ny*nz], xs[0:nmat*ng], mat[0:nx*ny*nz])
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < nx; i++)
            {
                for(unsigned int g = 0; g < ng; ++g)
                {
                    total_cross_section(g,i,j,k) = xs(mat(i,j,k)-1,g);
                }
            }
        }
    }

    STOP_PROFILING(__func__, true);
}

void calc_scattering_cross_section(void)
{
    START_PROFILING;

#pragma acc parallel \
    present(scat_cs[0:nmom*nx*ny*nz*ng], gg_cs[0:nmat*nmom*ng*ng], mat[0:nx*ny*nz])
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

    STOP_PROFILING(__func__, true);
}

// Calculate the outer source
void calc_outer_source(void)
{
    START_PROFILING;

//#pragma acc parallel \
    present(g2g_source[0:cmom*nx*ny*nz*ng], fixed_source[0:nx*ny*nz*ng],\
            scalar_flux[0:nx*ny*nz*ng], gg_cs[0:nmat*nmom*ng*ng], \
            mat[0:nx*ny*nz], scalar_mom[0:(cmom-1)*nx*ny*nz*ng])
    for (unsigned int g1 = 0; g1 < ng; g1++)
    {
        for(int ind = 0; ind < nx*ny*nz; ++ind)
        {
            int k = ind / (nx*ny);
            int j = (ind / nx) % ny;
            int i = ind % nx;

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
                    int lma_val = lma(l);
                    for (int m = 0; m < lma_val; m++)
                    {
                        g2g_source(mom+m,i,j,k,g1) += gg_cs(mat(i,j,k)-1,l,g2,g1) * scalar_mom(g2,mom+m-1,i,j,k);
                    }
                    mom += lma_val;
                }
            }
        }
    }

    STOP_PROFILING(__func__, true);
}

// Calculate the inner source
void calc_inner_source(void)
{
    START_PROFILING;

//#pragma acc parallel \
    present(source[0:cmom*nx*ny*nz*ng], g2g_source[0:cmom*nx*ny*nz*ng], \
            scat_cs[0:nmom*nx*ny*nz*ng], scalar_flux[0:nx*ny*nz*ng], \
            scalar_mom[0:(cmom-1)*nx*ny*nz*ng], lma[0:nmom])
    for(int k = 0; k < nz; ++k)
    {
        for(int j = 0; j < ny; ++j)
        {
            for(int i = 0; i < nx; ++i)
            {
                for (unsigned int g = 0; g < ng; g++)
                {
                    source(0,i,j,k,g) = g2g_source(0,i,j,k,g) 
                        + scat_cs(0,i,j,k,g) * scalar_flux(g,i,j,k);

                    // TODO: CHECK THIS IS CORRECT
                    unsigned int mom = 1;
                    for (unsigned int l = 1; l < nmom; l++)
                    {
                        int lma_val = lma(l);
                        for (int m = 0; m < lma_val; m++)
                        {
                            source(mom+m,i,j,k,g) = g2g_source(mom+m,i,j,k,g) 
                                + scat_cs(l,i,j,k,g) * scalar_mom(g,mom+m-1,i,j,k);
                        }
                        mom += lma_val;
                    }
                }
            }
        }
    }

    STOP_PROFILING(__func__, true);
}

void zero_flux_in_out(void)
{
//#pragma acc parallel \
    present(flux_in[0:nang*nx*ny*nz*ng*noct], flux_out[0:nang*nx*ny*nz*ng*noct])
    for(int i = 0; i < nang*nx*ny*nz*ng*noct; ++i)
    {
        flux_in[i] = 0.0;
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

//#pragma acc parallel \
    present(flux_i[0:fi_len], flux_j[0:fj_len], flux_k[0:fk_len])
    for(int i = 0; i < max_length; ++i)
    {
        if(i < fi_len) flux_i[i] = 0.0;
        if(i < fj_len) flux_j[i] = 0.0;
        if(i < fk_len) flux_k[i] = 0.0;
    }
}

void zero_flux_moments_buffer(void)
{
//#pragma acc parallel \
    present(scalar_mom[0:(cmom-1)*nx*ny*nz*ng])
    for(int i = 0; i < (cmom-1)*nx*ny*nz*ng; ++i)
    {
        scalar_mom[i] = 0.0;
    }
}

void zero_scalar_flux(void)
{
//#pragma acc parallel \
    present(scalar_flux[0:nx*ny*nz*ng])
    for(int i = 0; i < nx*ny*nz*ng; ++i)
    {
        scalar_flux[i] = 0.0;
    }
}

bool check_convergence(
        double *old, 
        double *new, 
        double epsi, 
        unsigned int *num_groups_todo, 
        int inner)
{
    START_PROFILING;
    int r = 1;
    int ngt = *num_groups_todo;

    // Reset the do_group list
    if (inner)
    {
        ngt = 0;
    }

//#pragma acc parallel private(r,ngt) \
    present(old[0:ng*nx*ny*nz], new[0:ng*nx*ny*nz], groups_todo[0:ng])
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

    *num_groups_todo = ngt;

    STOP_PROFILING(__func__, true);

    return r;
}

// Copies the value of scalar flux
void store_scalar_flux(double* to)
{
    START_PROFILING;

//#pragma acc parallel \
    present(to[0:nx*ny*nz*ng], scalar_flux[0:nx*ny*nz*ng])
    for(int i = 0; i < nx*ny*nz*ng; ++i)
    {
        to[i] = scalar_flux[i];
    }

    STOP_PROFILING(__func__, true);
}
