# Recipe of package ngsolve: NGSolve with netgen, plain CMake install.
# Sourced by stack.sh with STACK_NAME set to the software stack (empty for
# fetch). Interface: README.md, "Recipe interface".

VERSION=6.2.2608
CPUS=32
TIME=01:30:00
STACKS=(native-oss native-intel)
case ${STACK_NAME:-} in
native-oss) DEPENDS=(openblas) ;;
native-intel) DEPENDS=() ;;
*) DEPENDS=() ;;
esac

# fetch_ngsolve VERSION: login node, network only, into $STACK_SRC.
fetch_ngsolve() {
  stack_die "ngsolve $1: fetch not written yet (native-builds ticket 03)"
}

# build_ngsolve VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
build_ngsolve() {
  stack_die "ngsolve $1: build not written yet (native-builds ticket 05, 10)"
}

# smoke_ngsolve VERSION PREFIX: checks the install; records it with stack_smoke.
smoke_ngsolve() {
  stack_die "ngsolve $1: smoke not written yet (native-builds ticket 05, 10)"
}
