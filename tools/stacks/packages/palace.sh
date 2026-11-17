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

# palace_edit_mumps FILE TARBALL: the one source edit of the superbuild (ADR
# 0212, ADR 0205): its cmake/ExternalMUMPS.cmake line "-DMUMPS_gemmt=ON"
# becomes "-DMUMPS_gemmt=OFF" plus MUMPS_url naming the mirror's tarball
# (scivision's sub-configure downloads it otherwise; Palace forwards neither).
palace_edit_mumps() {
  local file=$1 tarball=$2 line='  "-DMUMPS_gemmt=ON"'
  [[ $(grep -cxF -- "$line" "$file") == 1 ]] ||
    stack_die "palace: $file has no single line $line: the MUMPS_gemmt/MUMPS_url edit needs updating"
  local edited="  \"-DMUMPS_gemmt=OFF\" \"-DMUMPS_url=$tarball\""
  LINE=$line EDITED=$edited awk '$0 == ENVIRON["LINE"] { print ENVIRON["EDITED"]; next } { print }' \
    "$file" >"$file.edited"
  mv "$file.edited" "$file"
}

# The palace-solver wheel's feature flags (wheelbuild/superbuild.py
# FEATURE_FLAGS), so wheel and native differ in toolchain, not in features.
# GPU dependencies off.
PALACE_FEATURES=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DPALACE_WITH_CUDA=OFF
  -DPALACE_WITH_HIP=OFF
  -DPALACE_WITH_MAGMA=OFF
  -DPALACE_WITH_GPU_AWARE_MPI=OFF
  -DPALACE_WITH_64BIT_INT=OFF
  -DPALACE_WITH_64BIT_BLAS_INT=OFF
  -DPALACE_WITH_OPENMP=ON
  -DPALACE_WITH_SUPERLU=ON
  -DPALACE_WITH_STRUMPACK=ON
  -DPALACE_WITH_STRUMPACK_BUTTERFLYPACK=OFF
  -DPALACE_WITH_STRUMPACK_ZFP=ON
  -DPALACE_WITH_MUMPS=ON
  -DPALACE_WITH_SLEPC=ON
  -DPALACE_WITH_ARPACK=ON
  -DPALACE_WITH_LIBXSMM=ON
  -DPALACE_WITH_GSLIB=ON
  -DPALACE_WITH_SUNDIALS=ON
)

# palace_edit_mumps FILE TARBALL: the one source edit of the superbuild (ADR
# 0212, ADR 0205): its cmake/ExternalMUMPS.cmake line "-DMUMPS_gemmt=ON"
# becomes "-DMUMPS_gemmt=OFF" plus MUMPS_url naming the mirror's tarball
# (scivision's sub-configure downloads it otherwise; Palace forwards neither).
palace_edit_mumps() {
  local file=$1 tarball=$2 line='  "-DMUMPS_gemmt=ON"'
  [[ $(grep -cxF -- "$line" "$file") == 1 ]] ||
    stack_die "palace: $file has no single line $line: the MUMPS_gemmt/MUMPS_url edit needs updating"
  local edited="  \"-DMUMPS_gemmt=OFF\" \"-DMUMPS_url=$tarball\""
  LINE=$line EDITED=$edited awk '$0 == ENVIRON["LINE"] { print ENVIRON["EDITED"]; next } { print }' \
    "$file" >"$file.edited"
  mv "$file.edited" "$file"
}

# palace_gemmt_imports FILE...: "FILE:SYMBOL" for each ?gemmt_ a shared object
# or executable imports (ADR 0205: MUMPS never calls GEMMT); nothing if none.
palace_gemmt_imports() {
  local file
  for file in "$@"; do
    nm -D --undefined-only "$file" |
      awk -v f="$file" '{ s = $NF; sub(/@.*/, "", s) } s ~ /^[sdcz]gemmt_$/ { print f ":" s }'
  done
}

# palace_offline_git FILE: the job-local git configuration (ADR 0212): every
# github.com, gitlab.com and bitbucket.org URL maps onto the source mirror, and
# file:// is allowed for submodules. Exported as GIT_CONFIG_GLOBAL, so
# ~/.gitconfig is never read or touched.
palace_offline_git() {
  local file=$1 host
  {
    for host in github.com gitlab.com bitbucket.org; do
      printf '[url "file://%s/mirrors/%s/"]\n\tinsteadOf = https://%s/\n' "$STACK_SRC" "$host" "$host"
    done
    printf '[protocol "file"]\n\tallow = always\n[advice]\n\tdetachedHead = false\n'
    printf '[user]\n\tname = stack.sh\n\temail = stack.sh@localhost\n'
  } >"$file"
  export GIT_CONFIG_GLOBAL=$file GIT_CONFIG_NOSYSTEM=1
}

# palace_cache CACHE VAR: the value of VAR in a CMakeCache.txt.
palace_cache() {
  sed -n "s/^$2:[A-Z]*=//p" "$1" | head -n 1
}

# palace_json_list ITEM...: the items as a JSON list.
palace_json_list() {
  "$STACK_PYTHON" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@"
}

# build_palace VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure,mumps,entry}.json.
# The superbuild at tag vVERSION, cloned from the mirror; its git dependencies
# through the job-local insteadOf, its URL tarballs through -DEXTERN_<NAME>_URL,
# its download_patch files pre-placed in the build tree, its MUMPS from the
# mirror's tarball without GEMMT. BLAS: the stack's OpenMP OpenBLAS
# (libopenblas_omp, ADR 0205), named to FindBLAS/FindLAPACK by their cache
# variables because BLA_VENDOR=OpenBLAS alone finds the pthreads libopenblas.
# The toolchain flags go in CMAKE_<LANG>_FLAGS only: libCEED's Makefile takes
# a CFLAGS environment variable instead of its own flags (and loses -fPIC).
# PALACE_TARGETS (probe only): build these superbuild targets, then fail.
build_palace() {
  local version=$1 prefix=$2 dir=$3 tag=v$1
  local src=$dir/palace build=$dir/build
  palace_offline_git "$dir/gitconfig"
  git clone -q --branch "$tag" "$PALACE_URL" "$src" || stack_die "palace: offline clone of $tag failed"
  local commit tags=$src/cmake/ExternalGitTags.cmake
  commit=$(git -C "$src" rev-parse HEAD)

  # Git dependencies (cloned by the superbuild through insteadOf): recorded at
  # the commits the mirror resolves their tags to.
  local name url ref line deps=() mumps_url= mumps_tag= metis_tag= scalapack_tag= lines=()
  mapfile -t lines < <(stack_sources palace-git <"$tags")
  for line in "${lines[@]}"; do
    read -r name url ref <<<"$line"
    [[ " ${PALACE_NOT_MIRRORED[*]} " != *" $name "* ]] || continue
    deps+=("$name:={\"url\": \"$url\", \"tag\": \"$ref\", \"commit\": \"$(
      git --git-dir="$(stack_mirror_dir "$url")" rev-parse "$ref^{commit}"
    )\"}")
    case $name in
    MUMPS) mumps_url=$url mumps_tag=$ref ;;
    METIS) metis_tag=$ref ;;
    SCALAPACK) scalapack_tag=$ref ;;
    esac
  done
  [[ -n $mumps_url && -n $metis_tag && -n $scalapack_tag ]] ||
    stack_die "palace $tag: no MUMPS, METIS or ScaLAPACK in ExternalGitTags.cmake"

  # URL dependencies: the mirror's tarballs (checked by fetch, trust on first use).
  local file args=() tarballs=()
  mapfile -t lines < <(stack_sources palace-tarballs <"$tags")
  for line in "${lines[@]}"; do
    read -r name file url <<<"$line"
    [[ -f $STACK_SRC/tarballs/$file ]] || stack_die "no tarballs/$file: run stack.sh fetch palace $version"
    args+=("-DEXTERN_${name}_URL=$STACK_SRC/tarballs/$file")
    tarballs+=("$name:={\"file\": \"$file\", \"sha256\": \"$(sha256sum "$STACK_SRC/tarballs/$file" | cut -d' ' -f1)\"}")
  done

  # MUMPS: the release its scivision tag names, checked against that tag's pin
  # (an explicit MUMPS_url turns scivision's own hash check off).
  local mumps_file pin mumps_tarball mumps_sha
  read -r mumps_file url pin < <(
    stack_show "$mumps_url" "$mumps_tag" cmake/source.json | stack_sources mumps-tarball "$mumps_tag"
  )
  mumps_tarball=$STACK_SRC/tarballs/$mumps_file
  [[ -f $mumps_tarball ]] || stack_die "no tarballs/$mumps_file: run stack.sh fetch palace $version"
  mumps_sha=$(sha256sum "$mumps_tarball" | cut -d' ' -f1)
  [[ $mumps_sha == "$pin" ]] || stack_die "palace: $mumps_file sha256 $mumps_sha is not scivision $mumps_tag's $pin"
  palace_edit_mumps "$src/cmake/ExternalMUMPS.cmake" "$mumps_tarball"

  # download_patch files, where the superbuild's configure looks for them.
  local patches=$STACK_SRC/patches/palace-$version
  [[ -d $patches ]] || stack_die "no $patches: run stack.sh fetch palace $version"
  mkdir -p "$build"
  cp -r "$patches/." "$build/"

  # BLAS/LAPACK of the stack, OpenMP threading for Palace.
  local blas_dir blas_lib
  case $STACK_BLAS_VENDOR in
  openblas)
    blas_dir=$(dirname "${STACK_OPENBLAS_PROVENANCE:?the openblas dependency is not activated}")
    blas_lib=$blas_dir/lib/lib$STACK_BLAS_LIB_OPENMP.so
    [[ -f $blas_lib ]] || stack_die "palace: no $blas_lib"
    args+=(-DBLA_VENDOR=OpenBLAS "-DBLAS_openblas_LIBRARY=$blas_lib" "-DLAPACK_openblas_LIBRARY=$blas_lib")
    export OPENBLAS_DIR=$blas_dir
    ;;
  *) stack_die "palace: BLAS vendor $STACK_BLAS_VENDOR is not written yet (native-builds ticket 10)" ;;
  esac
  # An MKL root would win the superbuild's vendor detection.
  unset MKLROOT MKL_DIR MKL_ROOT

  args=(
    "-DCMAKE_INSTALL_PREFIX=$prefix"
    "${PALACE_FEATURES[@]}"
    "-DCMAKE_C_FLAGS=$CFLAGS" "-DCMAKE_CXX_FLAGS=$CXXFLAGS" "-DCMAKE_Fortran_FLAGS=$FFLAGS"
    "-DCMAKE_INSTALL_RPATH=$prefix/lib;$prefix/lib64" -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
    "-DPALACE_GIT_COMMIT_ID=$tag" "-DPALACE_BUILD_ID=$STACK_NAME"
    "${args[@]}"
  )
  echo "palace: cmake ${args[*]}"
  env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS cmake -S "$src" -B "$build" "${args[@]}" ||
    stack_die "palace: superbuild configure failed"
  stack_meta configure "options:=$(palace_json_list "${args[@]}")" \
    "edit=cmake/ExternalMUMPS.cmake: -DMUMPS_gemmt=OFF -DMUMPS_url=$mumps_tarball" \
    "ldflags=$LDFLAGS" "blas_library=$blas_lib"
  stack_json "$dir/deps.json" "${deps[@]}"
  stack_json "$dir/tarballs.json" "${tarballs[@]}"
  stack_meta source "url=$PALACE_URL" "tag=$tag" "commit=$commit" \
    "git_dependencies:=$(<"$dir/deps.json")" "tarballs:=$(<"$dir/tarballs.json")" \
    "mumps_tarball:={\"file\": \"$mumps_file\", \"sha256\": \"$mumps_sha\", \"scivision_tag\": \"$mumps_tag\"}" \
    "patches=$patches"

  if [[ -n ${PALACE_TARGETS:-} ]]; then
    local targets=()
    read -ra targets <<<"$PALACE_TARGETS"
    env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS \
      cmake --build "$build" -j "$STACK_JOBS" --target "${targets[@]}" ||
      stack_die "palace: probe targets ${targets[*]} failed"
    stack_die "palace: probe done (PALACE_TARGETS=${targets[*]} built); no install"
  fi
  env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS cmake --build "$build" -j "$STACK_JOBS" ||
    stack_die "palace: superbuild failed"

  mkdir -p "$prefix/share/stack/palace"
  local cache
  for cache in CMakeCache.txt extern/mumps-build/CMakeCache.txt palace-build/CMakeCache.txt; do
    [[ ! -f $build/$cache ]] || cp "$build/$cache" "$prefix/share/stack/palace/${cache//\//_}"
  done
  palace_mumps_meta "$prefix" "$build/extern/mumps-build/CMakeCache.txt" \
    "$mumps_sha" "${scalapack_tag#v}" "${metis_tag##*-}"
}

# palace_mumps_meta PREFIX MUMPS_CACHE SHA256 SCALAPACK METIS: the provenance
# mumps block (ADR 0209, ticket 05's table) from the installed MUMPS.
palace_mumps_meta() {
  local prefix=$1 cache=$2 sha=$3 scalapack=$4 metis=$5
  local version intsize=32 fc a arith=() orderings=(amd pord) ordering libs=() gemmt
  version=$(sed -n 's/.*MUMPS_VERSION "\([0-9.]*\)".*/\1/p' "$prefix/include/dmumps_c.h" | head -n 1)
  [[ -n $version ]] || stack_die "palace: no MUMPS_VERSION in $prefix/include/dmumps_c.h"
  grep -q 'define MUMPS_INTSIZE64' "$prefix/include/mumps_int_def.h" && intsize=64
  fc=$(palace_cache "$cache" CMAKE_Fortran_COMPILER)
  for a in s d c z; do
    compgen -G "$prefix/lib*/lib${a}mumps.so" >/dev/null && arith+=("$a")
  done
  for ordering in metis parmetis scotch ptscotch; do
    [[ $(palace_cache "$cache" "MUMPS_$ordering") != ON ]] || orderings+=("$ordering")
  done
  mapfile -t orderings < <(printf '%s\n' "${orderings[@]}" | sort)
  mapfile -t libs < <(find "$prefix"/lib* -maxdepth 1 -type f \( -name 'lib*mumps*.so*' -o -name 'libpord*.so*' -o -name 'libpalace.so*' \) | sort)
  libs+=("$prefix"/bin/palace-*.bin)
  gemmt=$(palace_gemmt_imports "${libs[@]}")
  local openmp=false
  [[ $(palace_cache "$cache" MUMPS_openmp) != ON ]] || openmp=true
  stack_meta mumps "version=$version" "tarball_sha256=$sha" "intsize:=$intsize" \
    "gemmt:=$([[ -z $gemmt ]] && echo false || echo true)" \
    "gemmt_checked:=$(palace_json_list "${libs[@]##*/}")" \
    "blas:={\"vendor\": \"$STACK_BLAS_VENDOR\", \"version\": \"$STACK_BLAS_VERSION\", \"threading\": \"openmp\"}" \
    "scalapack=$scalapack" "compiler=$("$fc" --version | head -n 1)" \
    "fflags=$(palace_cache "$cache" CMAKE_Fortran_FLAGS)" \
    "arithmetics:=$(palace_json_list "${arith[@]}")" "openmp:=$openmp" \
    "orderings:=$(palace_json_list "${orderings[@]}")" "metis=$metis"
  [[ -z $gemmt ]] || stack_die "palace: MUMPS imports GEMMT (ADR 0205): $gemmt"
}

# smoke_palace VERSION PREFIX: checks the install; records it with stack_meta smoke.
# The one bin/palace-*.bin (recorded as the provenance entry the verdict
# runner launches) reports its version and dependencies under srun; it and
# MUMPS resolve BLAS to the stack's OpenMP OpenBLAS only (no pthreads
# libopenblas, no MKL, no libiomp5, nothing missing).
smoke_palace() {
  local version=$1 prefix=$2 bins=() bin out lib deps blas_dir
  bins=("$prefix"/bin/palace-*.bin)
  [[ ${#bins[@]} == 1 && -x ${bins[0]} ]] || stack_die "palace smoke: no single bin/palace-*.bin: ${bins[*]}"
  bin=${bins[0]}
  echo "\"bin/${bin##*/}\"" >"$STACK_META/entry.json"
  blas_dir=$(dirname "$STACK_OPENBLAS_PROVENANCE")
  for lib in "$bin" "$prefix"/lib*/libdmumps.so; do
    deps=$(ldd "$lib")
    [[ $deps != *"not found"* ]] || stack_die "palace smoke: ${lib##*/} has missing libraries: $deps"
    [[ $deps == *"lib$STACK_BLAS_LIB_OPENMP.so.0 => $blas_dir/lib/"* ]] ||
      stack_die "palace smoke: ${lib##*/} does not load $blas_dir/lib/lib$STACK_BLAS_LIB_OPENMP.so: $deps"
    ! grep -qE '^\s*(libopenblas\.so|libmkl|libiomp5)' <<<"$deps" ||
      stack_die "palace smoke: ${lib##*/} loads another BLAS or OpenMP runtime: $deps"
  done
  out=$(OMP_NUM_THREADS=1 srun --ntasks=1 --cpus-per-task=1 --mpi=pmix "$bin" --version 2>&1) ||
    stack_die "palace smoke: $bin --version failed: $out"
  [[ $out == *"$version"* ]] || stack_die "palace smoke: $bin --version does not report $version: $out"
  echo "palace smoke: $out"
  stack_meta smoke "command=srun --ntasks=1 --cpus-per-task=1 --mpi=pmix bin/${bin##*/} --version" \
    "output=$(head -n 1 <<<"$out")" "versions:=$(palace_json_list "${out}")" \
    "blas=$(ldd "$bin" | grep "lib$STACK_BLAS_LIB_OPENMP" | xargs)" "gemmt=none"
}
