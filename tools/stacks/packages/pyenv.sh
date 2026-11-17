# Recipe of package pyenv: the stack venv template (python-build-standalone) naming NGSolve and PETSc (ADR 0207).
# Sourced by stack.sh with STACK_NAME set to the software stack (empty for
# fetch). Interface: README.md, "Recipe interface".

VERSION=3.12.12
CPUS=16
TIME=01:00:00
STACKS=(native-oss native-intel)
case ${STACK_NAME:-} in
native-oss) DEPENDS=(ngsolve petsc) ;;
native-intel) DEPENDS=(ngsolve petsc) ;;
*) DEPENDS=() ;;
esac

# fetch_pyenv VERSION: login node, network only, into $STACK_SRC.
fetch_pyenv() {
  stack_die "pyenv $1: fetch not written yet (native-builds ticket 03)"
}

# build_pyenv VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
build_pyenv() {
  stack_die "pyenv $1: build not written yet (native-builds ticket 06, 10)"
}

# smoke_pyenv VERSION PREFIX: checks the install; records it with stack_smoke.
smoke_pyenv() {
  stack_die "pyenv $1: smoke not written yet (native-builds ticket 06, 10)"
}
