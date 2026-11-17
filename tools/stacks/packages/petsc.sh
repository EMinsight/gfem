# Recipe of package petsc: PETSc and SLEPc with petsc4py and slepc4py, gfem's ADR 0067 contract (ADR 0209).
# Sourced by stack.sh with STACK_NAME set to the software stack (empty for
# fetch). Interface: README.md, "Recipe interface".

VERSION=3.26.0
CPUS=32
TIME=01:30:00
STACKS=(native-oss native-intel)
case ${STACK_NAME:-} in
native-oss) DEPENDS=(openblas) ;;
native-intel) DEPENDS=() ;;
*) DEPENDS=() ;;
esac

# fetch_petsc VERSION: login node, network only, into $STACK_SRC.
fetch_petsc() {
  stack_die "petsc $1: fetch not written yet (native-builds ticket 03)"
}

# build_petsc VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
build_petsc() {
  stack_die "petsc $1: build not written yet (native-builds ticket 05, 10)"
}

# smoke_petsc VERSION PREFIX: checks the install; records it with stack_smoke.
smoke_petsc() {
  stack_die "petsc $1: smoke not written yet (native-builds ticket 05, 10)"
}
