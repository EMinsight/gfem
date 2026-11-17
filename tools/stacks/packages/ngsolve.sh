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

NGSOLVE_URL=https://github.com/NGSolve/ngsolve.git

# fetch_ngsolve VERSION: login node, network only, into $STACK_SRC.
# Read from tag vVERSION: mirrors of NGSolve and its submodules at the commits
# the tag records (netgen, then netgen's pybind11), the UMFPACK source of
# cmake/SuperBuild.cmake (BUILD_UMFPACK), and the OCCT wheels its
# pyproject.toml pins (netgen-occt, netgen-occt-devel: ticket 07 Q1).
fetch_ngsolve() {
  local tag=v$1 line url ref occt umfpack=
  local lines=()
  stack_mirror "$NGSOLVE_URL" "$tag"
  stack_mirror_submodules "$NGSOLVE_URL" "$tag"
  local repos
  repos=$(stack_show "$NGSOLVE_URL" "$tag" cmake/SuperBuild.cmake | stack_sources git-repositories)
  mapfile -t lines <<<"$repos"
  for line in "${lines[@]}"; do
    read -r url ref <<<"$line"
    [[ $url == */suitesparse-metis-for-windows.git ]] || continue
    stack_mirror "$url" "$ref"
    umfpack=$url
  done
  [[ -n $umfpack ]] || stack_die "ngsolve $tag: no suitesparse repository in cmake/SuperBuild.cmake"
  occt=$(stack_show "$NGSOLVE_URL" "$tag" pyproject.toml |
    grep -oE 'netgen-occt-devel==[0-9][0-9.]*' | head -n 1) ||
    stack_die "ngsolve $tag: pyproject.toml pins no netgen-occt-devel"
  stack_pip wheels "netgen-occt==${occt#*==}" "$occt"
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
