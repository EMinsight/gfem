# Toolchain of the native-intel software stack, MN5 GPP modules:
# GCC 12.3 for C, C++ and Fortran (not icx/ifx), MKL 2025.2 through mkl_rt
# with the GNU interface and threading layers, and Intel MPI 2021.16 over
# libfabric's mlx provider. "Intel" labels the closed BLAS and MPI only.
# ScaLAPACK 2.2.3 (not MKL's) and MUMPS 5.9.1 are built by each code's own
# build. No libiomp5 in any process.
#
# Sourced by `stack.sh build` (and frozen at the stack's first build as
# $STACKS_ROOT/native-intel/toolchain.sh) and by every activate.sh of the
# stack. The only place this stack's toolchain lives. Sets no thread counts.
export STACK_NAME=native-intel

# MN5's default environment loads intel, impi, mkl and ucx: start clean. One
# module per line: loading these three in one `module load` loads nothing.
module purge
module load gcc/12.3.0
module load impi/2021.16
module load mkl/2025.2
module load cmake/3.30.5

# Compilers: GCC 12.3 directly, and Intel MPI's GNU wrappers over it.
export CC=gcc CXX=g++ FC=gfortran F77=gfortran
export I_MPI_CC=gcc I_MPI_CXX=g++ I_MPI_FC=gfortran I_MPI_F90=gfortran I_MPI_F77=gfortran
export MPICC=mpicc MPICXX=mpicxx MPIFC=mpif90

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

# BLAS/LAPACK: MKL 2025.2 through mkl_rt, LP64. mkl_rt defaults to the Intel
# interface layer, whose complex return convention is not gfortran's, and to
# Intel threading (libiomp5): both layers set here, so configure tests and
# every process see them (activate.sh sources this file).
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export STACK_BLAS_VENDOR=mkl
export STACK_BLAS_VERSION=2025.2
export STACK_BLAS_LINK="$MKLROOT/lib/libmkl_rt.so"
export STACK_BLAS_THREADING="gnu (libgomp) for every code"
export STACK_SCALAPACK="reference 2.2.3, built by each code"

# MPI: Intel MPI 2021.16 over libfabric mlx; the module sets I_MPI_PMI_LIBRARY
# (Slurm's libpmi2) for srun launches.
export STACK_MPI_VENDOR="Intel MPI"
export STACK_MPI_VERSION=2021.16
export I_MPI_FABRICS=shm:ofi
export FI_PROVIDER=mlx
