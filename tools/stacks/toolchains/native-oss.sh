# Toolchain of the native-oss software stack (ADR 0205), MN5 GPP modules:
# GCC 12.3 for C, C++ and Fortran, Open MPI 4.1.6 over UCX 1.16, and the
# self-built OpenBLAS 0.3.34 of package openblas (pthreads libopenblas for
# gfem's NGSolve and PETSc, OpenMP libopenblas_omp for Palace). ScaLAPACK
# 2.2.3 and MUMPS 5.9.1 are built by each code's own build.
#
# Sourced by `stack.sh build` (and frozen at the stack's first build as
# $STACKS_ROOT/native-oss/toolchain.sh) and by every activate.sh of the stack.
# The only place this stack's toolchain lives. Sets no thread counts.
export STACK_NAME=native-oss

# MN5's default environment loads intel, impi, mkl and ucx: start clean.
module purge
module load gcc/12.3.0
module load ucx/1.16.0-gcc
module load openmpi/4.1.6-gcc
module load cmake/3.30.5

# Compilers: GCC 12.3 directly, and Open MPI's wrappers over it.
export CC=gcc CXX=g++ FC=gfortran F77=gfortran
export MPICC=mpicc MPICXX=mpicxx MPIFC=mpifort

# Flags: no fast-math, no LTO, no PGO, no -mprefer-vector-width=512.
# -mno-avx512fp16: MN5's binutils 2.35 cannot assemble the AVX512-FP16 (vmovw)
# that GCC 12.3 emits for sapphirerapids; double-precision codes lose nothing.
export CFLAGS="-O3 -march=sapphirerapids -mno-avx512fp16"
export CXXFLAGS="-O3 -march=sapphirerapids -mno-avx512fp16"
export FFLAGS="-O3 -march=sapphirerapids -mno-avx512fp16"
export FCFLAGS="-O3 -march=sapphirerapids -mno-avx512fp16"
# Recipes add this for Fortran packages that need it (MUMPS, ScaLAPACK).
export STACK_FORTRAN_COMPAT_FLAGS="-fallow-argument-mismatch"
# The system libstdc++ is GCC 11.5's: every shared library finds GCC 12.3's
# runtime by rpath (/apps/GPP/GCC/12.3.0/lib64 on MN5).
STACK_GCC_LIBDIR=$(dirname "$(readlink -f "$(gcc -print-file-name=libstdc++.so)")")
export STACK_GCC_LIBDIR
export LDFLAGS="-Wl,-rpath,$STACK_GCC_LIBDIR"

# BLAS/LAPACK: OpenBLAS 0.3.34 from package openblas, built with
# TARGET=SAPPHIRERAPIDS DYNAMIC_ARCH=0 NUM_THREADS=256 NO_AFFINITY=1; each
# code links the threading build of its own wheel.
export STACK_BLAS_VENDOR=openblas
export STACK_BLAS_VERSION=0.3.34
export STACK_BLAS_PACKAGE=openblas
export STACK_BLAS_LIB_PTHREADS=openblas
export STACK_BLAS_LIB_OPENMP=openblas_omp
export STACK_BLAS_LINK="-lopenblas (pthreads: NGSolve, PETSc) | -lopenblas_omp (OpenMP: Palace)"
export STACK_BLAS_THREADING="per code: pthreads (NGSolve, PETSc), openmp (Palace)"
export STACK_SCALAPACK="reference 2.2.3, built by each code"

# MPI: Open MPI 4.1.6, UCX point-to-point; launch with srun (PMI) or mpirun.
export STACK_MPI_VENDOR="Open MPI"
export STACK_MPI_VERSION=4.1.6
export OMPI_MCA_pml=ucx
export UCX_TLS=self,sm,dc_mlx5
