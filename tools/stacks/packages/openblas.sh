# Recipe of package openblas: OpenBLAS, pthreads libopenblas and OpenMP libopenblas_omp in one prefix (native-oss only).
# Sourced by stack.sh with STACK_NAME set to the software stack (empty for
# fetch). Interface: README.md, "Recipe interface".

VERSION=0.3.34
CPUS=16
TIME=01:00:00
STACKS=(native-oss)
case ${STACK_NAME:-} in
native-oss) DEPENDS=() ;;
native-intel) DEPENDS=() ;;
*) DEPENDS=() ;;
esac

# fetch_openblas VERSION: login node, network only, into $STACK_SRC.
# The release tarball; 0.3.34 is pinned by gfem's PETSc contract
# (tools/petsc_wheelhouse.sh TARBALLS), any other version is trusted on first
# fetch.
fetch_openblas() {
  local pin=
  case $1 in
  0.3.34) pin=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed ;;
  esac
  stack_tarball tarballs "OpenBLAS-$1.tar.gz" \
    "https://github.com/xianyi/OpenBLAS/archive/v$1.tar.gz" $pin
}

# build_openblas VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
build_openblas() {
  stack_die "openblas $1: build not written yet (native-builds ticket 04)"
}

# smoke_openblas VERSION PREFIX: checks the install; records it with stack_smoke.
smoke_openblas() {
  stack_die "openblas $1: smoke not written yet (native-builds ticket 04)"
}
