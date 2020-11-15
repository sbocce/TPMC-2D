TPMC-2D
===============================

A simple single-species Test-particle Monte Carlo (TPMC) algorithm in 2D 
(radial-axial plane), with hard-sphere gas-phase collisions and diffusive 
wall scattering.

This program comes with a GPL licence. 
If you use it for a scientific publication, you can cite:

"Numerical Investigation of Reversed Gas Feed Configurations for Hall Thrusters",
S. Boccelli, T.E. Magin, A. Frezzotti (submitted, 2020).

Description 
-------------------------------

You can find a detailed description of the algorithm in the previously mentioned 
paper (a preprint is available in this folder).

The algorithm injects particles from a given location, sampling them from a 
Maxwellian distribution at a given temperature and average velocity.
The particles are moved ballistically and are scattered when a wall is encountered. 
For a free-molecular flow this is all that one needs.

The presence of a background gas can be enabled as to introduce gas-phase collisions.
If so, the background gas is assumed Maxwellian and is imported as a 2D matrix, giving
the background density, average velocity (all three components) and temperature.

The residence time inside a given region is tracked and an average is printed for
all particles. 
You can easily save the residence time of all particles by 

Given its simplicity, this program is easily customizable to different geometries.

Description of the GPU version
-------------------------------

This program comes in two versions: MAIN\_CUDA.cu and MAIN\_SERIAL.cpp. 
The CUDA version runs on NVidia GPUs with CUDA capabilities.
On a GTX 760 it shows a speed-up of roughly 1600x with respect to the serial
version.
You can set single or double precision by uncommenting the proper typedef
in the very beginning of the code.

Building the source
-------------------------------

#### CPU version

The CPU version uses the Eigen library.
Compile with:
```
$ g++ main.cpp -I/usr/local/eigen3/
```

#### GPU version

You need to have installed CUDA for compiling this program.
The version of CUDA that you need depends on the compute capability of your GPU.
This program was tested for a compute capability = 3, on CUDA 10.2.

Once CUDA is installed, use the nvcc compiler:
```
nvcc -lcurand main.cu 
```
In case nvcc is not able to locate the librariues, just add them explicitly.
You can add the include directory with "-I" 
and the libraries directory with "-L".
For example:
```
nvcc -I/opt/nvidia/hpc_sdk/Linux_x86_64/2020/math_libs/include \
     -L/opt/nvidia/hpc_sdk/Linux_x86_64/2020/math_libs/lib64 \
     -lcurand main.cu
```

