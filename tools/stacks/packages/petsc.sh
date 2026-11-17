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

# The --download tarballs of gfem's PETSc contract, as tools/petsc_wheelhouse.sh
# TARBALLS pins them (ADR 0067, ADR 0209): "NAME URL [SHA256]". PETSc's
# configure cannot download through the tunnel (urllib), so the build passes
# --download-<pkg>=$STACK_SRC/tarballs/NAME. ParMETIS has no pin anywhere:
# trusted on first fetch (kept for ticket 05, gfem's stack builds without it).
PETSC_TARBALLS=(
  "MUMPS_5.9.1.tar.gz https://mumps-solver.org/MUMPS_5.9.1.tar.gz 659c9b57646b5a003ac618baa1faf9dd2044e46c732b3daaccbc7158003e1b46"
  "scalapack-v2.2.3.tar.gz https://github.com/Reference-ScaLAPACK/scalapack/archive/v2.2.3.tar.gz 5d93701eca663925e98010dd8d0f45fd79b2191d74e5afa5711d587370a8b9dd"
  "scotch-v7.0.18.tar.gz https://gitlab.inria.fr/scotch/scotch/-/archive/v7.0.18/scotch-v7.0.18.tar.gz 2cbdcf6da5a08ca31437d7c967b243b3c96d621409ecb710768de2aa2e09802d"
  "petsc-pkg-metis-v5.1.0-p12.tar.gz https://bitbucket.org/petsc/pkg-metis/get/v5.1.0-p12.tar.gz 2f16abe87394d4fd1430d66a8995076c352af40f7a4d1c5300b3b1cc9d545663"
  "petsc-pkg-parmetis-v4.0.3-p9.tar.gz https://bitbucket.org/petsc/pkg-parmetis/get/v4.0.3-p9.tar.gz"
)

# petsc_pin NAME VERSION: the sha256 tools/petsc_wheelhouse.sh pins for a
# release tarball (petsc, slepc, petsc4py, slepc4py), empty if none.
petsc_pin() {
  case $1-$2 in
  petsc-3.26.0) echo f5230023e6e22ee607802a13c82bc25f3d81e71654ad386a5b9bdff17fed93df ;;
  petsc4py-3.26.0) echo 0a34bd083e24bee1a6759c4639a45d10c0117855eb9f0466834450245f75d7e5 ;;
  slepc-3.26.0) echo a2f4cc2af76d55c078c30ad8bc66b44736dac921a7912266eb44136fc1b6029d ;;
  slepc4py-3.26.0) echo 07bf970c861736b9b977dfecd4ad46415d6c4a6f26b0ed128ecffae59613d6ff ;;
  esac
}

# fetch_petsc VERSION: login node, network only, into $STACK_SRC.
# PETSc and SLEPc release tarballs (the PyPI sdists, same bytes as the
# release snapshots), petsc4py/slepc4py sdists, the --download tarballs.
fetch_petsc() {
  local version=$1 name dir line url pin
  for name in petsc slepc petsc4py slepc4py; do
    dir=tarballs
    [[ $name != *4py ]] || dir=pip
    stack_tarball "$dir" "$name-$version.tar.gz" \
      "https://files.pythonhosted.org/packages/source/${name:0:1}/$name/$name-$version.tar.gz" \
      $(petsc_pin "$name" "$version")
  done
  for line in "${PETSC_TARBALLS[@]}"; do
    read -r name url pin <<<"$line"
    stack_tarball tarballs "$name" "$url" $pin
  done
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
