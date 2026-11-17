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

PALACE_URL=https://github.com/awslabs/palace.git
# Git dependencies of Palace's superbuild that a CPU build never clones
# (EXTERN_<NAME>_URL names): GPU only (MAGMA, Umpire) and tests only (Enzyme).
PALACE_NOT_MIRRORED=(MAGMA UMPIRE ENZYME)

# fetch_palace VERSION: login node, network only, into $STACK_SRC.
# Everything the superbuild at tag vVERSION fetches, read from that tag
# (ADR 0212): mirrors of its git dependencies (cmake/ExternalGitTags.cmake)
# and of Catch2 (test/unit/CMakeLists.txt, cloned at configure even without
# tests); its URL tarballs, trusted on first fetch; the MUMPS release that its
# scivision/mumps tag builds, pinned by that tag's cmake/source.json; and its
# download_patch files, pinned, under patches/palace-VERSION/<build-tree dir>.
fetch_palace() {
  local version=$1 tag=v$1 tags line name url ref file pin dir
  local lines=()
  stack_mirror "$PALACE_URL" "$tag"
  tags=$(stack_show "$PALACE_URL" "$tag" cmake/ExternalGitTags.cmake)

  local deps mumps_url= mumps_tag=
  deps=$(stack_sources palace-git <<<"$tags")
  mapfile -t lines <<<"$deps"
  for line in "${lines[@]}"; do
    read -r name url ref <<<"$line"
    [[ " ${PALACE_NOT_MIRRORED[*]} " != *" $name "* ]] || continue
    stack_mirror "$url" "$ref"
    [[ $name != MUMPS ]] || mumps_url=$url mumps_tag=$ref
  done
  [[ -n $mumps_url ]] || stack_die "palace $tag: no EXTERN_MUMPS_URL in ExternalGitTags.cmake"

  local catch2
  catch2=$(stack_show "$PALACE_URL" "$tag" test/unit/CMakeLists.txt | stack_sources git-repositories)
  mapfile -t lines <<<"$catch2"
  for line in "${lines[@]}"; do
    [[ -n $line ]] || continue
    read -r url ref <<<"$line"
    stack_mirror "$url" "$ref"
  done

  local tarballs
  tarballs=$(stack_sources palace-tarballs <<<"$tags")
  mapfile -t lines <<<"$tarballs"
  for line in "${lines[@]}"; do
    read -r name file url <<<"$line"
    stack_tarball tarballs "$file" "$url"
  done

  read -r file url pin < <(
    stack_show "$mumps_url" "$mumps_tag" cmake/source.json | stack_sources mumps-tarball "$mumps_tag"
  )
  [[ -n ${pin:-} ]] || stack_die "palace $tag: no MUMPS release for scivision $mumps_tag"
  stack_tarball tarballs "$file" "$url" "$pin"

  local cmake patches
  cmake=$(git --git-dir="$(stack_mirror_dir "$PALACE_URL")" ls-tree --name-only "$tag" cmake/)
  patches=$(
    for file in $cmake; do
      if [[ $file == *.cmake ]]; then
        stack_show "$PALACE_URL" "$tag" "$file"
      fi
    done | stack_sources patches
  )
  mapfile -t lines <<<"$patches"
  for line in "${lines[@]}"; do
    [[ -n $line ]] || continue
    read -r dir file url pin <<<"$line"
    stack_tarball "patches/palace-$version/$dir" "$file" "$url" "$pin"
  done
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
