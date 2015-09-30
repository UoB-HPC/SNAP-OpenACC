#pragma once

// This file contains a list of the global problem variables
// such as grid size, angles, energy groups, etc.
int nx;
int ny;
int nz;
int ng;
int nang;
int noct;
int cmom;
int nmom;
int nmat;

int ichunk;
int timesteps;

double dt;
double dx;
double dy;
double dz;

int outers;
int inners;

double epsi;
double tolr;

// Data
double* restrict source;
double* restrict flux_in;
double* restrict flux_out;
double* restrict flux_i;
double* restrict flux_j;
double* restrict flux_k;
double* restrict denom;
double dd_i;
double* restrict dd_j;
double* restrict dd_k;
double* restrict mu;
double* restrict eta;
double* restrict xi;
double* restrict scat_coeff;
double* restrict time_delta;
double* restrict total_cross_section;
double* restrict weights;
double* restrict velocity;
double* restrict scalar_flux;
double* restrict xs;
int* restrict mat;
double* restrict fixed_source;
double* restrict gg_cs;
int* restrict lma;
double* restrict g2g_source;
double* restrict scalar_mom;
double* restrict scat_cs;
unsigned int* restrict groups_todo;
double* restrict old_outer_scalar;
double* restrict old_inner_scalar;
double* restrict new_scalar;

// Global variable for the timestep
unsigned int global_timestep;
