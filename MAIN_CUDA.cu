// Copyright (c) 2020, Stefano Boccelli
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice, 
//   this list of conditions and the following disclaimer in the documentation 
//   and/or other materials provided with the distribution.
// * Neither the name of the copyright holder's organization nor the names of 
//   its contributors may be used to endorse or promote products derived from 
//   this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL S. BOCCELLI BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
////////////////////////////////////////////////////////////////////////////////
//
// You can find a description of the algorithm in:
// "Numerical Investigation of Reversed Gas Feed Configurations for Hall 
//  Thrusters", S. Boccelli, T.E. Magin, A. Frezzotti (submitted, 2020).
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <string>

#include <cuda.h>
#include <curand_kernel.h>

#define PI 3.1415926535
#define KB 1.38064852E-23

// typedef double fpTYPE; // Floating point type. Defined here for fast switching from double to float in case of GPU.
typedef float fpTYPE; // Floating point type. Defined here for fast switching from double to float in case of GPU.

#include "ReadFiles_cuda.h"


// ############################################################################

__global__ void initRNG_kernel(curandState *state)
{
  int th_id = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(12345, th_id, 0, &state[th_id]);
}

// ############################################################################

__device__ double rnum(curandState *state)
{
  int th_id = threadIdx.x + blockIdx.x*blockDim.x;

  // return ((double) rand() / (RAND_MAX));
  return curand_uniform_double(&state[th_id]);
}

// ############################################################################

__device__ void maxwellian(fpTYPE UX, fpTYPE UY, fpTYPE UZ, fpTYPE T, 
                           fpTYPE M, curandState *state, fpTYPE* vel_new)
{
  // Computes three velocities from a drifted Maxwellian

  for(size_t i = 0; i < 3; ++i)
  {
    // Step 1.
    fpTYPE THETA = 2.0*PI*rnum(state);

    // Step 2.
    fpTYPE BETA = 1.0/sqrt(2.0*KB/M*T);

    // R may go from 0 to 1 included. Remove the extremes in a very rough way
    // or the log() will explode
    vel_new[i] = sin(THETA)*sqrt(-log(rnum(state)*0.9999999 + 1.0e-10))/BETA;
  }

  // Step 3. Add average velocity
  vel_new[0] += UX;
  vel_new[1] += UY;
  vel_new[2] += UZ;

  return;

}

// #####################################################

__device__ fpTYPE psi_HS(fpTYPE x)
{
// This is an auxiliary function for the computation of HS collisions 
// (see Cercignani, The Boltzmann equation and its applications (1988), pag. 179).

  return exp(-x*x) + (2.0*x + 1.0/(x + 0.000000001))*sqrt(PI)/2.0*erf(x);
}

// #####################################################

__device__ void interp2D(fpTYPE xP, fpTYPE yP, 
                         fpTYPE* xx, 
                         fpTYPE* yy,
                         fpTYPE* N_MAT, 
                         fpTYPE* V1_MAT, 
                         fpTYPE* V2_MAT, 
                         fpTYPE* V3_MAT, 
                         fpTYPE* T_MAT,
                         size_t Nx, size_t Ny,
                         fpTYPE *outparam)
{

  // xP is x3 in the paper reference frame;
  // yP is R in the paper reference frame.

  // i1  is the first index before the particle location, along x
  // j1  is the first index before the particle location, along y

  fpTYPE Lx = xx[Nx-1] - xx[0]; 
  fpTYPE Ly = yy[Ny-1] - yy[0]; 

  size_t i = floor((xP - xx[0]) / Lx * Nx) + 1;
  size_t j = floor((yP - yy[0]) / Ly * Ny) + 1;

  // Make sure the index is in the range (maybe particle went crazy!)
  if(i < 1)   {
    i = 0;
  } else if(i >= Nx-1)   {
    i = Nx-2;
  } 

  if(j < 1)  {
    j = 0;
  } else if(j >= Ny-1)   {
    j = Ny-2;
  }

  // Find the vector-ID for nodes
  size_t id_a = (i+1)*Ny + (j+1);
  size_t id_b = (i+1)*Ny + j;
  size_t id_c = (i)*Ny + j;
  size_t id_d = (i)*Ny + (j+1);

  // Compute weights
  fpTYPE A = (xP - xx[i])*(yP - yy[j]);
  fpTYPE B = (xP - xx[i])*(yy[j+1] - yP);
  fpTYPE C = (xx[i+1] - xP)*(yy[j+1] - yP);
  fpTYPE D = (xx[i+1] - xP)*(yP - yy[j]);

  // Interpolation
  outparam[0] = (A*N_MAT[id_a]  + B*N_MAT[id_b]  + C*N_MAT[id_c]  + D*N_MAT[id_d])  / (A+B+C+D); // Number density
  outparam[1] = (A*V1_MAT[id_a] + B*V1_MAT[id_b] + C*V1_MAT[id_c] + D*V1_MAT[id_d]) / (A+B+C+D); // V1 (v_R)
  outparam[2] = (A*V2_MAT[id_a] + B*V2_MAT[id_b] + C*V2_MAT[id_c] + D*V2_MAT[id_d]) / (A+B+C+D); // V2 (v_theta)
  outparam[3] = (A*V3_MAT[id_a] + B*V3_MAT[id_b] + C*V3_MAT[id_c] + D*V3_MAT[id_d]) / (A+B+C+D); // V3 (v_z)
  outparam[4] = (A*T_MAT[id_a]  + B*T_MAT[id_b]  + C*T_MAT[id_c]  + D*T_MAT[id_d])  / (A+B+C+D); // Temperature

  return;
}

// ##############################################################
 
__device__ void collision_rotate_vel(fpTYPE& v1, fpTYPE& v2, fpTYPE& v3, fpTYPE* v_collpart, fpTYPE alpha, curandState *state)
{
  // Compute relative velocity
  fpTYPE g1 = v_collpart[0] - v1;
  fpTYPE g2 = v_collpart[1] - v2;
  fpTYPE g3 = v_collpart[2] - v3;

  fpTYPE g = sqrt(g1*g1 + g2*g2 + g3*g3);

  // Perform collision!
  fpTYPE q = 2.0*pow(rnum(state), 1.0/alpha) - 1.0;
  fpTYPE cos_th = q;
  fpTYPE sin_th = sqrt(1.0 - q*q);
  
  fpTYPE chi = 2*PI*rnum(state);
  fpTYPE cos_chi = cos(chi);
  fpTYPE sin_chi = sin(chi);
  
  fpTYPE g1_prime = g*sin_th*cos_chi;
  fpTYPE g2_prime = g*sin_th*sin_chi;
  fpTYPE g3_prime = g*cos_th;

  v1 = v_collpart[0] + g1_prime;
  v2 = v_collpart[1] + g2_prime;
  v3 = v_collpart[2] + g3_prime;

  return;
}

// ############################################################################

__global__ void myKernel(curandState *state, int pPERt, size_t Nx_BG, size_t Ny_BG, 
                         fpTYPE* d_xx, fpTYPE* d_yy, 
                         fpTYPE* d_N_BG, fpTYPE* d_V1_BG, fpTYPE* d_V2_BG, fpTYPE* d_V3_BG, fpTYPE* d_T_BG,
                         fpTYPE* d_times,
                         fpTYPE* d_x1P, fpTYPE* d_x2P, fpTYPE* d_x3P,
                         fpTYPE  z_slit_start)
{

  fpTYPE x1, x2, x3;
  fpTYPE v1, v2, v3;
  fpTYPE v1_prime, v2_prime;
  fpTYPE R, th;

  // Simulation parameters
  fpTYPE Ldomain  = 0.025;   // [m] length of domain (channel)
  fpTYPE R_chan_ext = 0.05;  // [m] radius of external wall
  fpTYPE R_chan_int = 0.035; // [m] radius of internal wall

  fpTYPE Twall  = 300.0; // [K]
  fpTYPE Tanode = 300.0; // [K]
  fpTYPE Tinj   = 300.0; // [K]
  fpTYPE M     = 2.18e-25;  // [kg] particles mass

  // // Uniform injection from anode 
  // fpTYPE u1_inj = 0.0; // uR
  // fpTYPE u2_inj = 0.0; // utheta
  // fpTYPE u3_inj = 0.0; // uz
 
  // Injection from slit 
  fpTYPE u1_inj = 88.975; // uR
  fpTYPE u2_inj = 0.0; // utheta
  fpTYPE u3_inj = -154.11; // uz

  int tID_glob = blockIdx.x*blockDim.x + threadIdx.x;

  for(size_t IDp = 0; IDp < pPERt; ++IDp)
  {
    // Unique identifier of particle
    size_t pID_global = IDp + tID_glob*pPERt;
    d_times[pID_global] = 0.0; // Init

    // // Inject particles from anode
    // x1 = rnum(state)*(R_chan_ext - R_chan_int)*0.98 + R_chan_int*1.001;
    // x2 = 0.0;
    // x3 = 0.0;

    // Inject particles from side walls
    x1 = R_chan_int;
    x2 = 0.0;
    x3 = rnum(state)*0.001 + z_slit_start; // 1 mm slit
  
    fpTYPE vel_now[3];
    maxwellian(u1_inj, u2_inj, u3_inj, Tinj, M, state, vel_now);
  
    v1 = vel_now[0];
    v2 = vel_now[1];
    v3 = vel_now[2];
  
    // Loop until particle stays inside the domain
    size_t counter = 0;
    while( x3 < Ldomain )
    {
      // std::cout << "Advecting..." << std::endl;
  
      // Timestep adjustment
  
      fpTYPE v_abs = sqrt(v1*v1 + v2*v2 + v3*v3); 
      fpTYPE dt = (Ldomain/(v_abs + 1.0e-5))/50.0; // so it takes 100 steps to do one full domain
  
      // Timestep adjustment
  
      // ++++++++ Advect particle ++++++++++++
      x1 += v1*dt;
      x2 += v2*dt;
      x3 += v3*dt;
  
      // Rotate back position into plane (cylindrical coordinates)
      R  = sqrt(x1*x1 + x2*x2);
      th = atan2(x2,x1); 
  
      x1 = R;
      x2 = 0;
  
      // Rotate velocity
      v1_prime = cos(th)*v1 + sin(th)*v2;
      v2_prime = -sin(th)*v1 + cos(th)*v2;
  
      v1 = v1_prime; // Updated velocity
      v2 = v2_prime;
  
      // +++++++++ Check boundaries ++++++++++
      if( R < R_chan_int ) // Inner wall hit
      {
        fpTYPE tOUT = abs( (x1 - R_chan_int)/(v1+1.0e-10) ); // How much time was spent out (spend it IN!)
  
        // Remove trajectory out of wall
        x1 -= tOUT*v1;
        x3 -= tOUT*v3;
  
        v1 = -1; // Init like this to enter the loop
        while (v1 < 0)
        {
          maxwellian(0.0, 0.0, 0.0, Twall, M, state, vel_now);
          v1 = vel_now[0];
          v2 = vel_now[1];
          v3 = vel_now[2];
        }
     
        // Add trajectory out of wall
        // x1 = R_chan_int + v1*tOUT; // Finish advection
        x1 += v1*tOUT;
        x3 += v3*tOUT;
  
      } 
      else if (R > R_chan_ext) // Outer wall hit
      {
        fpTYPE tOUT = abs( (x1 - R_chan_ext)/(v1+1.0e-10) ); // How much time was spent out (spend it IN!)
  
        // Remove trajectory out of wall
        x1 -= tOUT*v1;
        x3 -= tOUT*v3;
  
        v1 = +1; // Init like this to enter the loop
        while (v1 > 0)
        {
          maxwellian(0.0, 0.0, 0.0, Twall, M, state, vel_now);
          v1 = vel_now[0];
          v2 = vel_now[1];
          v3 = vel_now[2];
        }
      
        // Add trajectory out of wall
        // x1 = R_chan_ext + v1*tOUT; // Finish advection
        x1 += v1*tOUT;
        x3 += v3*tOUT;
  
      }
      else if (x3 < 0.0) // Back wall hit
      {
        fpTYPE tOUT = abs( x3/(v3+1.0e-10) ); // How much time was spent out (spend it IN!)
  
        // Remove trajectory out of wall
        x1 -= v1*tOUT;
        x3 -= v3*tOUT;
  
        v3 = -1; // Init like this to enter the loop
        while (v3 < 0)
        {
          maxwellian(0.0, 0.0, 0.0, Tanode, M, state, vel_now);
          v1 = vel_now[0];
          v2 = vel_now[1];
          v3 = vel_now[2];
        }
  
        // Add trajectory out of wall
        // x3 = v3*tOUT; // Finish advection
        x1 += v1*tOUT;
        x3 += v3*tOUT;
      }

      // ++++++++ Perform collisions +++++++++++

      // Interpolate parameters at local position
      fpTYPE params[5];
      interp2D(x3, R, d_xx, d_yy, d_N_BG, d_V1_BG, d_V2_BG, d_V3_BG, d_T_BG, Nx_BG, Ny_BG, params);
  
      fpTYPE n_bg_now  = params[0];
      fpTYPE u1_bg_now = params[1];
      fpTYPE u2_bg_now = params[2];
      fpTYPE u3_bg_now = params[3];
      fpTYPE T_bg_now  = params[4];

      // ***** Hard-Sphere cross section *****
      fpTYPE d_molec = 5.74e-10;
      fpTYPE sig = PI*d_molec*d_molec; // [m2] Cross section
      fpTYPE alpha = 1.0;  // Scattering parameter (Bird, 1994, Appendix A) = 1 for isotropic scattering
      fpTYPE vTH_rel = sqrt(8*KB*T_bg_now/PI/(M/2.0)); // relative thermal vel: use reduced mass!

      fpTYPE g1 = v1 - u1_bg_now; // Relative velocity
      fpTYPE g2 = v2 - u2_bg_now; // Relative velocity
      fpTYPE g3 = v3 - u3_bg_now; // Relative velocity

      fpTYPE nu = sig*n_bg_now/PI*sqrt(2*PI*KB*T_bg_now/M)*psi_HS(sqrt(g1*g1 + g2*g2 + g3*g3)*sqrt(M/(2*KB*T_bg_now)) );
      // SIMPLE CASE // fpTYPE nu = vTH_rel*n_bg_now*sig;// collision frequency
      // *****************************
      
      fpTYPE Pcoll = 1 - exp(-nu*dt);
  
      if ( rnum(state) < Pcoll ) // Collision happens?
      { 
        // Create colliding neutral from local features
        fpTYPE v_collpart[3];
        maxwellian(u1_bg_now, u2_bg_now, u3_bg_now, T_bg_now, M, state, v_collpart);

        // Perform collision
        collision_rotate_vel(v1, v2, v3, v_collpart, alpha, state); 
      }

      // DBDBDB - export some particles, for testing purposes
      d_x1P[counter] = x1;
      d_x2P[counter] = x2;
      d_x3P[counter] = x3;
      counter++;

      // Add timestep to particle residence time
      if( (x3 <= 0.018) && (x3 >= 0.013) ) {
        d_times[pID_global] += dt; // Add timestep
      }

    } // end while particle inside domain

  } // end loop on particles

}

// ############################################################################

int main()
{
  int NTH = 512; // number of threads per block
  // int NTH = 84; // number of threads per block
  int NB  = 60; // number of blocks
  int pPERt = 33; // particles per thread

  // // For testing
  // int NTH   = 1;
  // int NB    = 1;
  // int pPERt = 1;

  // std::cout << "We will simulate " << pPERt*NTH*NB << " particles." << std::endl;

  // ========= Setup PRNG on GPU ========================
  curandState *devStates;
  cudaMalloc((void **)&devStates, NB*NTH*sizeof(curandState));

  // ========= Load background gas data =================
  fpTYPE z_slit_start = 0.015;
  
  // std::string dirname = "matrices_uniform/";
  std::string dirname = "matrices_rev30_15_COLD/";

  size_t Nx_BG, Ny_BG;
  std::ifstream f( (dirname + "MAT_N.dat").c_str() );  // Read first line of a random file
  f >> Nx_BG >> Ny_BG; // Read dimensions as first elements, (Nx  Ny)

  // Allocate host memory
  fpTYPE *h_xx = new fpTYPE[Nx_BG];
  fpTYPE *h_yy = new fpTYPE[Ny_BG];

  fpTYPE *h_N_BG  = new fpTYPE[Nx_BG*Ny_BG];
  fpTYPE *h_V1_BG = new fpTYPE[Nx_BG*Ny_BG];
  fpTYPE *h_V2_BG = new fpTYPE[Nx_BG*Ny_BG];
  fpTYPE *h_V3_BG = new fpTYPE[Nx_BG*Ny_BG];
  fpTYPE *h_T_BG  = new fpTYPE[Nx_BG*Ny_BG];

  // Load background gas data into host arrays
  FillVectorFile(h_xx,    (dirname + "xx.dat").c_str());
  FillVectorFile(h_yy,    (dirname + "yy.dat").c_str());
  FillMatrixFile(h_N_BG,  (dirname + "MAT_N.dat").c_str());
  FillMatrixFile(h_V1_BG, (dirname + "MAT_V.dat").c_str()); // "y" in the DSMC simulation
  FillMatrixFile(h_V2_BG, (dirname + "MAT_W.dat").c_str()); // "z" in the DSMC simulation
  FillMatrixFile(h_V3_BG, (dirname + "MAT_U.dat").c_str()); // "x" in the DSMC simulation
  FillMatrixFile(h_T_BG,  (dirname + "MAT_T.dat").c_str());

  // Init device variables
  fpTYPE *d_xx; 
  fpTYPE *d_yy; 

  fpTYPE *d_N_BG; 
  fpTYPE *d_V1_BG;
  fpTYPE *d_V2_BG;
  fpTYPE *d_V3_BG;
  fpTYPE *d_T_BG; 

  cudaMalloc(&d_xx, sizeof(fpTYPE)*Nx_BG);
  cudaMalloc(&d_yy, sizeof(fpTYPE)*Ny_BG);

  cudaMalloc(&d_N_BG,  sizeof(fpTYPE)*Nx_BG*Ny_BG);
  cudaMalloc(&d_V1_BG, sizeof(fpTYPE)*Nx_BG*Ny_BG);
  cudaMalloc(&d_V2_BG, sizeof(fpTYPE)*Nx_BG*Ny_BG);
  cudaMalloc(&d_V3_BG, sizeof(fpTYPE)*Nx_BG*Ny_BG);
  cudaMalloc(&d_T_BG,  sizeof(fpTYPE)*Nx_BG*Ny_BG);

  // Copy stuff into GPU global memory
  cudaMemcpy(d_xx, h_xx, sizeof(fpTYPE)*Nx_BG, cudaMemcpyHostToDevice);
  cudaMemcpy(d_yy, h_yy, sizeof(fpTYPE)*Ny_BG, cudaMemcpyHostToDevice);

  cudaMemcpy(d_N_BG,  h_N_BG,  sizeof(fpTYPE)*Nx_BG*Ny_BG, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V1_BG, h_V1_BG, sizeof(fpTYPE)*Nx_BG*Ny_BG, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V2_BG, h_V2_BG, sizeof(fpTYPE)*Nx_BG*Ny_BG, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V3_BG, h_V3_BG, sizeof(fpTYPE)*Nx_BG*Ny_BG, cudaMemcpyHostToDevice);
  cudaMemcpy(d_T_BG,  h_T_BG,  sizeof(fpTYPE)*Nx_BG*Ny_BG, cudaMemcpyHostToDevice);

  // ========= Call kernel =================
  fpTYPE *h_times = new fpTYPE[NB*NTH*pPERt]; // Total number of particles
  fpTYPE *d_times;
  cudaMalloc(&d_times, sizeof(fpTYPE)*NB*NTH*pPERt);

  size_t Ntest = 100000;
  fpTYPE *h_x1P = new fpTYPE[Ntest];
  fpTYPE *h_x2P = new fpTYPE[Ntest];
  fpTYPE *h_x3P = new fpTYPE[Ntest];

  fpTYPE *d_x1P; 
  fpTYPE *d_x2P; 
  fpTYPE *d_x3P; 

  cudaMalloc(&d_x1P, sizeof(fpTYPE)*Ntest);
  cudaMalloc(&d_x2P, sizeof(fpTYPE)*Ntest);
  cudaMalloc(&d_x3P, sizeof(fpTYPE)*Ntest);

  // for(size_t slitID = 0; slitID < 24; slitID++) {
  //   fpTYPE z_slit_start = 0.024  - slitID*0.001;

    initRNG_kernel<<<NB,NTH>>>(devStates);
    myKernel<<<NB,NTH>>>(devStates, pPERt, Nx_BG, Ny_BG, d_xx, d_yy, d_N_BG, d_V1_BG, d_V2_BG, d_V3_BG, d_T_BG, d_times, d_x1P, d_x2P, d_x3P, z_slit_start);
    
    cudaMemcpy(h_times,  d_times,  sizeof(fpTYPE)*NB*NTH*pPERt, cudaMemcpyDeviceToHost);
  
    // // Print residence times now
    // for(size_t IDp = 0; IDp < NB*NTH*pPERt; ++IDp)
    // {
    //   std::cout << h_times[IDp] << std::endl;
    // }
  
    // Compute average residence time
    fpTYPE tau_ave = 0.0;
    
    for(size_t IDp = 0; IDp < NB*NTH*pPERt; ++IDp)
    {
      tau_ave += h_times[IDp]/(NB*NTH*pPERt);
    }
    
    std::cout << z_slit_start << "  " << tau_ave << std::endl;

  // }

  // cudaMemcpy(h_x1P,  d_x1P,  sizeof(fpTYPE)*Ntest, cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_x2P,  d_x2P,  sizeof(fpTYPE)*Ntest, cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_x3P,  d_x3P,  sizeof(fpTYPE)*Ntest, cudaMemcpyDeviceToHost);

  // for (size_t iii = 0; iii < Ntest; ++iii)
  //   std::cout << h_x1P[iii] << " " << h_x2P[iii] << " " << h_x3P[iii] << std::endl;

  // ======== Freed memory =================
  cudaFree(devStates);

  cudaFree(d_xx);
  cudaFree(d_yy);

  cudaFree(d_N_BG); 
  cudaFree(d_V1_BG);
  cudaFree(d_V2_BG);
  cudaFree(d_V3_BG);
  cudaFree(d_T_BG); 
  
  delete[] h_xx;
  delete[] h_yy;

  delete[] h_N_BG; 
  delete[] h_V1_BG;
  delete[] h_V2_BG;
  delete[] h_V3_BG;
  delete[] h_T_BG; 

  cudaFree(d_times);
  delete[] h_times;

  // TESTING 
  cudaFree(d_x1P); 
  cudaFree(d_x2P); 
  cudaFree(d_x3P); 
  delete[] h_x1P;
  delete[] h_x2P;
  delete[] h_x3P;


  return 0;
}
