# sphexa-analysis

This is an anlysis pipeline for simulations created with the [sphexa](https://github.com/sphexa-org/sphexa) simulation code.

# Usage


# Installation

First you need to load the neccessary modules on the supercomputer of your choice. On DAINT use

```shell
uenv start prgenv-gnu/24.11:v1 --view=default
```

#### HEFFTE installation

This Pipeline relies on the [HEFFTE](https://github.com/af-ayala/heffte) library for performing fourier transforms. It is therefore neccessary to install this library first. If you have it installed already, you can skip this step.

```shell
git clone https://github.com/af-ayala/heffte.git
cd heffte
mkdir build
cd build
CC=mpicc CCX=mpicxx cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_FLAGS=-ccbin=mpicxx -DCMAKE_CUDA_ARCHITECTURES=90 -D BUILD_SHARED_LIBS=ON -D CMAKE_INSTALL_PREFIX=<lib-path> -D HEFFTE_ENABLE_FFTW=ON -D HEFFTE_ENABLE_CUDA=ON ..
make -j10
make install
```
add the library to your PATH variable by adding 
```shell
export PATH=<lib-path>
```
in your .bashrc file. Keep in mind, that HEFFTE requires FFTW to install correctly. MKL is bugged and has been for ages. You also need both the single and double precision version of FFTW even though we only use the double precision version.

#### Installation of the analysis code

When Heffte is available, install the analysis code using the following commands:


on DAINT:
```shell
git clone https://github.com/olavril/sphexa-analysis.git
cd sphexa-analysis
mkdir build
cd build
CC
make sphexa-analysis
cp main/src/sphexa_analysis <executable-path>
```

on HELIX:
```shell
git clone https://github.com/olavril/sphexa-analysis.git
cd sphexa-analysis
mkdir build
cd build
CC=mpicc CXX=mpicxx cmake -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_FLAGS=-ccbin=mpicxx -S ..
make sphexa-analysis
cp main/src/sphexa_analysis <executable-path>
```
