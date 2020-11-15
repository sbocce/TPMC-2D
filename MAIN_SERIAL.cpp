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
#include <string>
#include <math.h>
#include <vector>
#include <time.h>
#include <Eigen/Dense>
 
#include "ReadFiles.h"

#define PI 3.1415926535
#define KB 1.38064852E-23

using namespace Eigen;

// typedef double fpTYPE; // Floating point type. Defined here for fast switching from double to float in case of GPU.
typedef float fpTYPE; // Floating point type. Defined here for fast switching from double to float in case of GPU.

// ############################################################################

fpTYPE rnum()
{
  return ((fpTYPE) rand() / (RAND_MAX));
}

// ############################################################################

void maxwellian(fpTYPE UX, fpTYPE UY, fpTYPE UZ, fpTYPE T, fpTYPE M, fpTYPE* vel_new)
{
  // Computes three velocities from a drifted Maxwellian

  for(size_t i = 0; i < 3; ++i)
  {
    // Step 1.
    fpTYPE THETA = 2.0*3.14159265359*rnum();

    // Step 2.
    fpTYPE BETA = 1.0/sqrt(2.0*1.38064852e-23/M*T);

    // R may go from 0 to 1 included. Remove the extremes in a very rough way
    // or the log() will explode
    vel_new[i] = sin(THETA)*sqrt(-log(rnum()*0.9999999 + 1.0e-10))/BETA;
  }

  // Step 3. Add average velocity
  vel_new[0] += UX;
  vel_new[1] += UY;
  vel_new[2] += UZ;

  return;

}

// ############################################################################

void interp2D(fpTYPE xP, fpTYPE yP, Eigen::VectorXd xx, Eigen::VectorXd yy, Eigen::MatrixXd N_MAT, 
              Eigen::MatrixXd V1_MAT, Eigen::MatrixXd V2_MAT, Eigen::MatrixXd V3_MAT, 
              Eigen::MatrixXd T_MAT, fpTYPE *outparam)
{

  // xP is x3 in the paper reference frame;
  // yP is R in the paper reference frame.

  // i1  is the first index before the particle location, along x
  // j1  is the first index before the particle location, along y

  size_t Nx = xx.size();
  size_t Ny = yy.size();

  fpTYPE Lx = xx(Nx-1) - xx(0); 
  fpTYPE Ly = yy(Ny-1) - yy(0); 

  size_t i1 = floor((xP - xx(0)) / Lx * Nx) + 1;
  size_t j1 = floor((yP - yy(0)) / Ly * Ny) + 1;

  // Make sure the index is in the range (maybe particle went crazy!)
  if(i1 < 1) 
  {
    i1 = 1;
  } 
  else if(i1 > Nx-1) 
  {
    i1 = Nx-1;
  }

 
  if(j1 < 1)
  {
    j1 = 1;
  }
  else if(j1 > Ny-1)
  {
    j1 = Ny-1;
  }

  outparam[0] = N_MAT(i1,j1);  // Number density
  outparam[1] = V1_MAT(i1,j1); // V1 (V_R)
  outparam[2] = V2_MAT(i1,j1); // V1 (V_theta)
  outparam[3] = V3_MAT(i1,j1); // V3 (V_z)
  outparam[4] = T_MAT(i1,j1);  // Temperature

  return;
}

// ##############################################################
 
void collision_HS(fpTYPE& v1, fpTYPE& v2, fpTYPE& v3, fpTYPE* v_collpart)
{
  // Compute relative velocity
  fpTYPE g1 = v_collpart[0] - v1;
  fpTYPE g2 = v_collpart[1] - v2;
  fpTYPE g3 = v_collpart[2] - v3;

  fpTYPE g = sqrt(g1*g1 + g2*g2 + g3*g3);

  // Perform collision!
  fpTYPE q = 2.0*rnum() - 1.0;
  fpTYPE cos_th = q;
  fpTYPE sin_th = sqrt(1.0 - q*q);
  
  fpTYPE chi = 2*PI*rnum();
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

int main()
{
  srand (time(NULL)); // TMP!!!

  size_t Np = 1000;  // Number of particles
  std::vector< fpTYPE > v_res_times(Np, 0.0); // vector of residence times


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

  // Injection velocity (Maxwellian)
  fpTYPE u1_inj = 88.975; 
  fpTYPE u2_inj = 0.0;
  fpTYPE u3_inj = -154.11;

  // +++++++++++++ Background gas: ++++++++++++++
  Eigen::VectorXd xx, yy;

  Eigen::MatrixXd N_BG, V1_BG, V2_BG, V3_BG, T_BG;

  // Read files and fill matrices

  FillVectorFile(xx, "matrices/xx.dat");
  FillVectorFile(yy, "matrices/yy.dat");

  FillMatrixFile(N_BG,  "matrices/MAT_N.dat");
  FillMatrixFile(V1_BG, "matrices/MAT_V.dat"); // "y" in the DSMC simulation
  FillMatrixFile(V2_BG, "matrices/MAT_W.dat"); // "z" in the DSMC simulation
  FillMatrixFile(V3_BG, "matrices/MAT_U.dat"); // "x" in the DSMC simulation
  FillMatrixFile(T_BG,  "matrices/MAT_T.dat");

  // --- Parameters for collisions 
  fpTYPE d_molec = 5.74e-10;
  fpTYPE sig = PI*d_molec*d_molec; // [m2] Cross section

  // fpTYPE nu = 4*nBG*sig/3.1415*sqrt(nBG*1.38e-23*Tlocal/M); //  TO BE MODIFIED

  // +++++++++++ TIME LOOP +++++++++++++

  for(size_t IDp = 0; IDp < Np; ++IDp)
  {
    //std::cout << "Processing particle " << IDp << " of " << Np-1 << std::endl;

    // // Inject particle from anode
    // // x1 = rnum()*(R_chan_ext - R_chan_int)*0.98 + R_chan_int*1.001; // Uniform injection
    // x1 = rnum()*0.001 + 0.042; // Slit injection
    // x2 = 0.0;
    // x3 = 0.0;

    // // Inject particles from side walls
    x1 = R_chan_int;
    x2 = 0.0;
    x3 = rnum()*0.001 + 0.015; // 1 mm slit

    fpTYPE vel_now[3];
    maxwellian(u1_inj, u2_inj, u3_inj, Tinj, M, vel_now);

    v1 = vel_now[0];
    v2 = vel_now[1];
    v3 = vel_now[2];

    fpTYPE t_inside = 0.0;

    // Loop until particle stays inside the domain
    while( x3 < Ldomain )
    {
      // std::cout << "Advecting..." << std::endl;

      // std::cout << x1 << " " << x2 << " " << x3 << std::endl;

      // Timestep adjustment
      fpTYPE v_abs = sqrt(v1*v1 + v2*v2 + v3*v3); 
      fpTYPE dt = (Ldomain/(v_abs + 1.0e-5))/50.0; // so it takes 100 steps to do one full domain

      // Update time spent inside
      t_inside += dt;

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
          maxwellian(0.0, 0.0, 0.0, Twall, M, vel_now);
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
          maxwellian(0.0, 0.0, 0.0, Twall, M, vel_now);
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
          maxwellian(0.0, 0.0, 0.0, Tanode, M, vel_now);
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
      interp2D(x3, R, xx, yy, N_BG, V1_BG, V2_BG, V3_BG, T_BG, params);
  
      fpTYPE n_bg_now  = params[0];
      fpTYPE u1_bg_now = params[1];
      fpTYPE u2_bg_now = params[2];
      fpTYPE u3_bg_now = params[3];
      fpTYPE T_bg_now  = params[4];

      fpTYPE vTH_rel = sqrt(8*KB*T_bg_now/PI/(M/2.0)); // relative thermal vel: use reduced mass!
      fpTYPE nu = vTH_rel*n_bg_now*sig;// collision frequency
  
      fpTYPE Pcoll = 1 - exp(-nu*dt);
      // fpTYPE coll_happens = (rnum() < Pcoll); 

      // std::cout << "Pc: " << Pcoll << " - coll happens? " << coll_happens << "  - nu: " << nu << "  -  n: " << n_bg_now << "  - T: " << T_bg_now << "  - vth: " << vTH << "  - sig: " << sig <<  std::endl;
 
//CHECK nu CHECK nu
//CHECK nu CHECK nu
//CHECK nu CHECK nu
//CHECK nu CHECK nu
//CHECK nu CHECK nu
//CHECK nu CHECK nu

      if ( rnum() < Pcoll ) // Collision happens?
      { 
        // Create colliding neutral from local features
        fpTYPE v_collpart[3];
        maxwellian(u1_bg_now, u2_bg_now, u3_bg_now, T_bg_now, M, v_collpart);

        // Perform collision
        collision_HS(v1, v2, v3, v_collpart); 
      }
  
      // std::cout << x1 << " " << x2 << " " << x3 << std::endl;
    }
    //  //// std::cout << "Particle ID: " << IDp << " - T spent inside: " << t_inside << std::endl;
    std::cout << t_inside << std::endl;
    //   //// v_res_times[IDp] = t_inside;

  }

  return 0;
}
