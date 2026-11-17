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
fetch_openblas() {
  stack_die "openblas $1: fetch not written yet (native-builds ticket 03)"
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
