# Recipe of package palace: Palace superbuild, offline, MUMPS without GEMMT.
# Sourced by stack.sh with STACK_NAME set to the software stack (empty for
# fetch). Interface: README.md, "Recipe interface".

VERSION=0.19.0
CPUS=56
TIME=03:00:00
STACKS=(native-oss native-intel)
case ${STACK_NAME:-} in
native-oss) DEPENDS=(openblas) ;;
native-intel) DEPENDS=() ;;
*) DEPENDS=() ;;
esac

# fetch_palace VERSION: login node, network only, into $STACK_SRC.
fetch_palace() {
  stack_die "palace $1: fetch not written yet (native-builds ticket 03)"
}

# build_palace VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
build_palace() {
  stack_die "palace $1: build not written yet (native-builds ticket 09, 10)"
}

# smoke_palace VERSION PREFIX: checks the install; records it with stack_smoke.
smoke_palace() {
  stack_die "palace $1: smoke not written yet (native-builds ticket 09, 10)"
}
