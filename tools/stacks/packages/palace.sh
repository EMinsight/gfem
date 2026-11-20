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
# Everything the superbuild at tag vVERSION fetches, read from that tag:
# mirrors of its git dependencies (cmake/ExternalGitTags.cmake)
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
    # What the tarball's own CMake files clone when it configures (scnlib:
    # fast_float), mirrored for the job's insteadOf.
    local nested nested_lines=() nested_line
    nested=$(stack_sources archive-git "$STACK_SRC/tarballs/$file")
    mapfile -t nested_lines <<<"$nested"
    for nested_line in "${nested_lines[@]}"; do
      [[ -n $nested_line ]] || continue
      read -r url ref <<<"$nested_line"
      stack_mirror "$url" "$ref"
    done
  done

  read -r file url pin < <(palace_mumps_release "$mumps_url" "$mumps_tag")
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
  # v0.18.1 downloads no patch; the build copies the directory either way.
  mkdir -p "$STACK_SRC/patches/palace-$version"
}

# palace_mumps_pin VERSION: sha256 of a MUMPS release that a scivision commit
# without cmake/source.json builds (Palace v0.18.1 pins one: MUMPS 5.7.3, as
# the palace-solver 0.18.1 wheel). Digests from scivision/mumps v5.9.1.2's
# cmake/source.json "mumps_sha256".
palace_mumps_pin() {
  case $1 in
  5.7.3) echo 84a47f7c4231b9efdf4d4f631a2cae2bdd9adeaabc088261d15af040143ed112 ;;
  *) stack_die "palace: no sha256 pin for MUMPS $1 (add it to palace_mumps_pin)" ;;
  esac
}

# palace_mumps_release URL REF: "FILE URL SHA256" of the MUMPS release that
# scivision/mumps at REF builds: from its cmake/source.json (tags v5.8 on),
# else from its CMakeLists.txt's MUMPS_UPSTREAM_VERSION and palace_mumps_pin.
palace_mumps_release() {
  local url=$1 ref=$2 file release
  if git --git-dir="$(stack_mirror_dir "$url")" cat-file -e "$ref:cmake/source.json" 2>/dev/null; then
    stack_show "$url" "$ref" cmake/source.json | stack_sources mumps-tarball "$ref"
    return
  fi
  read -r file release < <(stack_show "$url" "$ref" CMakeLists.txt | stack_sources mumps-release)
  [[ -n ${release:-} ]] || stack_die "palace: scivision $ref names no MUMPS release"
  local version=${file#MUMPS_}
  echo "$file $release $(palace_mumps_pin "${version%.tar.gz}")"
}

# palace_edit_mumps FILE TARBALL: the one source edit of the superbuild,
# so it builds offline from the mirror: its cmake/ExternalMUMPS.cmake line "-DMUMPS_gemmt=ON"
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

# palace_edit_mumps_gemmt_option FILE: the superbuilds whose ExternalMUMPS.cmake
# has no MUMPS_gemmt line (v0.18.1) build a scivision tag that reads option
# "gemmt" (default ON): its line "-DMUMPS_parallel=ON" gains "-Dgemmt=OFF".
palace_edit_mumps_gemmt_option() {
  local file=$1 line='  "-DMUMPS_parallel=ON"'
  [[ $(grep -cxF -- "$line" "$file") == 1 ]] ||
    stack_die "palace: $file has no single line $line: the gemmt edit needs updating"
  LINE=$line EDITED="$line \"-Dgemmt=OFF\"" \
    awk '$0 == ENVIRON["LINE"] { print ENVIRON["EDITED"]; next } { print }' "$file" >"$file.edited"
  mv "$file.edited" "$file"
}

# palace_mumps_url_edit TREE TARBALL: in a scivision/mumps tree without
# MUMPS_url (v5.7.3.1-26), its one download URL line names the mirror's
# tarball (FetchContent_Populate downloads it otherwise).
palace_mumps_url_edit() {
  local file=$1/CMakeLists.txt
  local line='set(url "https://mumps-solver.org/MUMPS_${MUMPS_UPSTREAM_VERSION}.tar.gz")'
  [[ $(grep -cxF -- "$line" "$file") == 1 ]] ||
    stack_die "palace: $file has no single line $line: the MUMPS url edit needs updating"
  LINE=$line EDITED="set(url \"$2\")" \
    awk '$0 == ENVIRON["LINE"] { print ENVIRON["EDITED"]; next } { print }' "$file" >"$file.edited"
  mv "$file.edited" "$file"
}

# palace_mumps_patch URL REF PATCH TARBALL DIR: Palace's own MUMPS patch file
# (git apply'd on the scivision checkout) regenerated, in DIR/mumps-patch,
# as that patch plus palace_mumps_url_edit, against scivision at REF.
palace_mumps_patch() {
  local url=$1 ref=$2 patch=$3 tarball=$4 tree=$5/mumps-patch
  git clone -q "$(stack_mirror_dir "$url")" "$tree" && git -C "$tree" checkout -q "$ref" ||
    stack_die "palace: no scivision $ref in the mirror"
  git -C "$tree" apply "$patch" || stack_die "palace: $patch does not apply to scivision $ref"
  palace_mumps_url_edit "$tree" "$tarball"
  git -C "$tree" diff >"$patch.new" && [[ -s $patch.new ]] || stack_die "palace: empty MUMPS patch"
  mv "$patch.new" "$patch"
}

# palace_edit_blas_forward FILE: Palace v0.18.1's top CMakeLists.txt passes
# no BLAS choice to its sub-builds, so their FindBLAS (MUMPS, SUNDIALS) finds
# the pthreads libopenblas beside the OpenMP one on CMAKE_PREFIX_PATH: the
# default sub-build arguments gain the FindBLAS/FindLAPACK cache variables the
# recipe sets (palace_blas_args; empty on an MKL stack).
palace_edit_blas_forward() {
  local file=$1 line='  "-DCMAKE_STATIC_LINKER_FLAGS=${CMAKE_STATIC_LINKER_FLAGS}"'
  [[ $(grep -cxF -- "$line" "$file") == 1 ]] ||
    stack_die "palace: $file has no single line $line: the BLAS forward edit needs updating"
  LINE=$line awk '
    $0 == ENVIRON["LINE"] {
      print
      print "  \"-DBLAS_openblas_LIBRARY=${BLAS_openblas_LIBRARY}\""
      print "  \"-DLAPACK_openblas_LIBRARY=${LAPACK_openblas_LIBRARY}\""
      next
    }
    { print }' "$file" >"$file.edited"
  mv "$file.edited" "$file"
}

# palace_edit_pkg_targets FILE: Palace v0.18.1's palace/CMakeLists.txt links
# STRUMPACK's optional packages by bare name ("-lzfp", and zfp installs to
# lib64): the line becomes upstream's fix (eb94e7af2, in v0.19.0), the
# imported target <pkg>::<pkg> when it exists. Nothing to do without the line.
palace_edit_pkg_targets() {
  local file=$1 line='      target_link_libraries(${LIB_TARGET_NAME} PUBLIC ${pkg})'
  local count
  count=$(grep -cxF -- "$line" "$file" || true)
  [[ $count != 0 ]] || return 0
  [[ $count == 1 ]] || stack_die "palace: $file has $count lines $line: the package-target edit needs updating"
  LINE=$line awk '
    $0 == ENVIRON["LINE"] {
      print "      if(TARGET ${pkg}::${pkg})"
      print "        target_link_libraries(${LIB_TARGET_NAME} PUBLIC ${pkg}::${pkg})"
      print "      else()"
      print "        target_link_libraries(${LIB_TARGET_NAME} PUBLIC ${pkg})"
      print "      endif()"
      next
    }
    { print }' "$file" >"$file.edited"
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

# palace_blas_lib: the BLAS/LAPACK library Palace links on this stack: the
# OpenMP OpenBLAS of package openblas (native-oss), mkl_rt (native-intel,
# GNU layers from the toolchain's MKL_*_LAYER at run time).
palace_blas_lib() {
  local lib
  case ${STACK_BLAS_VENDOR:-} in
  openblas)
    lib=$(dirname "${STACK_OPENBLAS_PROVENANCE:?the openblas dependency is not activated}")/lib/lib$STACK_BLAS_LIB_OPENMP.so
    ;;
  mkl) lib=${MKLROOT:?STACK_BLAS_VENDOR=mkl but no MKLROOT}/lib/libmkl_rt.so ;;
  *) stack_die "palace: unknown STACK_BLAS_VENDOR '${STACK_BLAS_VENDOR:-}'" ;;
  esac
  [[ -f $lib ]] || stack_die "palace: no $lib"
  echo "$lib"
}

# palace_blas_args: the superbuild's BLAS options, one per line. OpenBLAS is
# named to FindBLAS/FindLAPACK by their cache variables because
# BLA_VENDOR=OpenBLAS alone finds the pthreads libopenblas; MKL is the single
# dynamic library (Intel10_64_dyn = mkl_rt), never MKL's ScaLAPACK.
palace_blas_args() {
  local lib
  lib=$(palace_blas_lib) || return 1
  case $STACK_BLAS_VENDOR in
  openblas) printf '%s\n' -DBLA_VENDOR=OpenBLAS "-DBLAS_openblas_LIBRARY=$lib" "-DLAPACK_openblas_LIBRARY=$lib" ;;
  mkl) echo -DBLA_VENDOR=Intel10_64_dyn ;;
  esac
}

# palace_srun_mpi: srun's MPI plugin for the stack's MPI: PMIx for Open MPI,
# PMI-2 for Intel MPI (the module's I_MPI_PMI_LIBRARY is Slurm's libpmi2).
palace_srun_mpi() {
  case ${STACK_MPI_VENDOR:-} in
  "Open MPI") echo --mpi=pmix ;;
  "Intel MPI") echo --mpi=pmi2 ;;
  *) stack_die "palace: no srun MPI plugin for STACK_MPI_VENDOR '${STACK_MPI_VENDOR:-}'" ;;
  esac
}

# palace_gemmt_imports FILE...: "FILE:SYMBOL" for each ?gemmt_ a shared object
# or executable imports (MUMPS must never call GEMMT); nothing if none.
palace_gemmt_imports() {
  local file
  for file in "$@"; do
    nm -D --undefined-only "$file" |
      awk -v f="$file" '{ s = $NF; sub(/@.*/, "", s) } s ~ /^[sdcz]gemmt_$/ { print f ":" s }'
  done
}

# palace_offline_git FILE: the job-local git configuration: every
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


# build_palace VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure,mumps,entry}.json.
# The superbuild at tag vVERSION, cloned from the mirror; its git dependencies
# through the job-local insteadOf, its URL tarballs through -DEXTERN_<NAME>_URL,
# its download_patch files pre-placed in the build tree, its MUMPS from the
# mirror's tarball without GEMMT. BLAS: the stack's OpenMP OpenBLAS
# (libopenblas_omp), named to FindBLAS/FindLAPACK by their cache
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
    deps+=("$name:=$(stack_json - "url=$url" "tag=$ref" "commit=$(
      git --git-dir="$(stack_mirror_dir "$url")" rev-parse "$ref^{commit}"
    )")")
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
    tarballs+=("$name:=$(stack_json - "file=$file" "sha256=$(prov sha256 "$STACK_SRC/tarballs/$file")")")
  done

  # MUMPS: the release its scivision tag names, checked against that tag's pin
  # (an explicit MUMPS_url turns scivision's own hash check off).
  local mumps_file pin mumps_tarball mumps_sha edit
  read -r mumps_file url pin < <(palace_mumps_release "$mumps_url" "$mumps_tag")
  mumps_tarball=$STACK_SRC/tarballs/$mumps_file
  [[ -f $mumps_tarball ]] || stack_die "no tarballs/$mumps_file: run stack.sh fetch palace $version"
  mumps_sha=$(prov sha256 "$mumps_tarball")
  [[ $mumps_sha == "$pin" ]] || stack_die "palace: $mumps_file sha256 $mumps_sha is not scivision $mumps_tag's $pin"
  if grep -qxF -- '  "-DMUMPS_gemmt=ON"' "$src/cmake/ExternalMUMPS.cmake"; then
    palace_edit_mumps "$src/cmake/ExternalMUMPS.cmake" "$mumps_tarball"
    edit="cmake/ExternalMUMPS.cmake: -DMUMPS_gemmt=OFF -DMUMPS_url=$mumps_tarball"
  else
    # v0.18.1: scivision's options of that time, Palace's single MUMPS patch file.
    local mumps_patch=$src/extern/patch/mumps/patch_build.diff
    grep -qxF -- '  "${CMAKE_SOURCE_DIR}/extern/patch/mumps/patch_build.diff"' "$src/cmake/ExternalMUMPS.cmake" ||
      stack_die "palace: ExternalMUMPS.cmake applies another MUMPS patch than $mumps_patch"
    palace_edit_mumps_gemmt_option "$src/cmake/ExternalMUMPS.cmake"
    palace_mumps_patch "$mumps_url" "$mumps_tag" "$mumps_patch" "$mumps_tarball" "$dir"
    palace_edit_blas_forward "$src/CMakeLists.txt"
    edit="cmake/ExternalMUMPS.cmake: -Dgemmt=OFF; extern/patch/mumps/patch_build.diff: + set(url $mumps_tarball); CMakeLists.txt: sub-builds get BLAS_openblas_LIBRARY/LAPACK_openblas_LIBRARY"
  fi
  if grep -qxF -- '      target_link_libraries(${LIB_TARGET_NAME} PUBLIC ${pkg})' "$src/palace/CMakeLists.txt"; then
    palace_edit_pkg_targets "$src/palace/CMakeLists.txt"
    edit="$edit; palace/CMakeLists.txt: STRUMPACK packages as <pkg>::<pkg> (upstream eb94e7af2)"
  fi

  # download_patch files, where the superbuild's configure looks for them.
  local patches=$STACK_SRC/patches/palace-$version
  [[ -d $patches ]] || stack_die "no $patches: run stack.sh fetch palace $version"
  mkdir -p "$build"
  cp -r "$patches/." "$build/"

  # BLAS/LAPACK of the stack, OpenMP threading for Palace.
  local blas_lib blas_args=()
  mapfile -t blas_args < <(palace_blas_args) && [[ ${#blas_args[@]} -gt 0 ]] ||
    stack_die "palace: no BLAS for $STACK_NAME"
  blas_lib=$(palace_blas_lib)
  args+=("${blas_args[@]}")
  case $STACK_BLAS_VENDOR in
  openblas)
    export OPENBLAS_DIR=${blas_lib%/lib/*}
    # An MKL root would win the superbuild's vendor detection.
    unset MKLROOT MKL_DIR MKL_ROOT
    ;;
  mkl)
    # MKLROOT stays: the superbuild takes MKL_DIR from it (PETSc's MKL headers).
    unset OPENBLAS_DIR
    ;;
  esac

  args=(
    "-DCMAKE_INSTALL_PREFIX=$prefix"
    "${PALACE_FEATURES[@]}"
    "-DCMAKE_C_FLAGS=$CFLAGS" "-DCMAKE_CXX_FLAGS=$CXXFLAGS" "-DCMAKE_Fortran_FLAGS=$FFLAGS"
    "-DCMAKE_INSTALL_RPATH=$prefix/lib;$prefix/lib64" -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
    "-DPALACE_GIT_COMMIT_ID=$tag" "-DPALACE_BUILD_ID=$STACK_NAME"
    "${args[@]}"
  )
  # Palace configure runs PETSc/MPI test programs without a launcher; Intel MPI
  # singletons abort while I_MPI_PMI_LIBRARY (Slurm libpmi2) is set.
  echo "palace: cmake ${args[*]}"
  stack_singleton env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS cmake -S "$src" -B "$build" "${args[@]}" ||
    stack_die "palace: superbuild configure failed"
  stack_meta configure "options:=$(stack_json_array "${args[@]}")" \
    "edit=$edit" \
    "ldflags=$LDFLAGS" "blas_library=$blas_lib"
  stack_json "$dir/deps.json" "${deps[@]}"
  stack_json "$dir/tarballs.json" "${tarballs[@]}"
  stack_meta source "url=$PALACE_URL" "tag=$tag" "commit=$commit" \
    "git_dependencies:=$(<"$dir/deps.json")" "tarballs:=$(<"$dir/tarballs.json")" \
    "mumps_tarball:=$(stack_json - "file=$mumps_file" "sha256=$mumps_sha" "scivision_tag=$mumps_tag")" \
    "patches=$patches"

  if [[ -n ${PALACE_TARGETS:-} ]]; then
    local targets=()
    read -ra targets <<<"$PALACE_TARGETS"
    stack_singleton env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS \
      cmake --build "$build" -j "$STACK_JOBS" --target "${targets[@]}" ||
      stack_die "palace: probe targets ${targets[*]} failed"
    stack_die "palace: probe done (PALACE_TARGETS=${targets[*]} built); no install"
  fi
  stack_singleton env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS cmake --build "$build" -j "$STACK_JOBS" ||
    stack_die "palace: superbuild failed"

  mkdir -p "$prefix/share/stack/palace"
  local cache
  for cache in CMakeCache.txt extern/mumps-build/CMakeCache.txt palace-build/CMakeCache.txt; do
    [[ ! -f $build/$cache ]] || cp "$build/$cache" "$prefix/share/stack/palace/${cache//\//_}"
  done
  palace_mumps_meta "$prefix" "$build/extern/mumps-build/CMakeCache.txt" \
    "$mumps_sha" "${scalapack_tag#v}" "${metis_tag##*-}"
}

# palace_mumps_orderings CACHE: the orderings of a MUMPS build, one per line,
# sorted: AMD and PORD always, plus each ordering option ON in its CMake cache,
# named MUMPS_<ordering> (scivision v5.8 on) or <ordering> (v5.7, Palace 0.18.1).
palace_mumps_orderings() {
  local cache=$1 ordering orderings=(amd pord)
  for ordering in metis parmetis scotch ptscotch; do
    if [[ $(palace_cache "$cache" "MUMPS_$ordering") == ON || $(palace_cache "$cache" "$ordering") == ON ]]; then
      orderings+=("$ordering")
    fi
  done
  printf '%s\n' "${orderings[@]}" | sort
}

# palace_mumps_meta PREFIX MUMPS_CACHE SHA256 SCALAPACK METIS: the provenance
# mumps block from the installed MUMPS.
palace_mumps_meta() {
  local prefix=$1 cache=$2 sha=$3 scalapack=$4 metis=$5
  local version intsize=32 fc a arith=() orderings=() libs=() gemmt
  version=$(sed -n 's/.*MUMPS_VERSION "\([0-9.]*\)".*/\1/p' "$prefix/include/dmumps_c.h" | head -n 1)
  [[ -n $version ]] || stack_die "palace: no MUMPS_VERSION in $prefix/include/dmumps_c.h"
  grep -q 'define MUMPS_INTSIZE64' "$prefix/include/mumps_int_def.h" && intsize=64
  fc=$(palace_cache "$cache" CMAKE_Fortran_COMPILER)
  for a in s d c z; do
    compgen -G "$prefix/lib*/lib${a}mumps.so" >/dev/null && arith+=("$a")
  done
  mapfile -t orderings < <(palace_mumps_orderings "$cache")
  mapfile -t libs < <(find "$prefix"/lib* -maxdepth 1 -type f \( -name 'lib*mumps*.so*' -o -name 'libpord*.so*' -o -name 'libpalace.so*' \) | sort)
  libs+=("$prefix"/bin/palace-*.bin)
  gemmt=$(palace_gemmt_imports "${libs[@]}")
  local openmp=false threading=openmp
  [[ $(palace_cache "$cache" MUMPS_openmp) != ON ]] || openmp=true
  [[ $STACK_BLAS_VENDOR != mkl ]] || threading=mkl_rt
  stack_meta mumps "version=$version" "tarball_sha256=$sha" "intsize:=$intsize" \
    "gemmt:=$([[ -z $gemmt ]] && echo false || echo true)" \
    "gemmt_checked:=$(stack_json_array "${libs[@]##*/}")" \
    "blas:={\"vendor\": \"$STACK_BLAS_VENDOR\", \"version\": \"$STACK_BLAS_VERSION\", \"threading\": \"$threading\"}" \
    "scalapack=$scalapack" "compiler=$("$fc" --version | head -n 1)" \
    "fflags=$(palace_cache "$cache" CMAKE_Fortran_FLAGS)" \
    "arithmetics:=$(stack_json_array "${arith[@]}")" "openmp:=$openmp" \
    "orderings:=$(stack_json_array "${orderings[@]}")" "metis=$metis"
  [[ -z $gemmt ]] || stack_die "palace: MUMPS imports GEMMT: $gemmt"
}

# smoke_palace VERSION PREFIX: checks the install; records it with stack_meta smoke.
# The one bin/palace-*.bin (recorded as the provenance entry the verdict
# runner launches) reports its version and dependencies under srun (the
# plugin of the stack's MPI); it and MUMPS resolve BLAS to the stack's only:
# the OpenMP OpenBLAS (no pthreads libopenblas, no MKL) or the toolchain's
# mkl_rt (no OpenBLAS, no MKL ScaLAPACK/BLACS: reference ScaLAPACK from the
# prefix); never libiomp5, nothing missing.
smoke_palace() {
  local version=$1 prefix=$2 bins=() bin out lib deps blas_lib blas_line srun_mpi
  bins=("$prefix"/bin/palace-*.bin)
  [[ ${#bins[@]} == 1 && -x ${bins[0]} ]] || stack_die "palace smoke: no single bin/palace-*.bin: ${bins[*]}"
  bin=${bins[0]}
  echo "\"bin/${bin##*/}\"" >"$STACK_META/entry.json"
  blas_lib=$(palace_blas_lib)
  case $STACK_BLAS_VENDOR in
  openblas) blas_line="lib$STACK_BLAS_LIB_OPENMP.so.0 => $(dirname "$blas_lib")/" ;;
  mkl) blas_line="libmkl_rt.so.2 => $(dirname "$blas_lib")/" ;;
  esac
  for lib in "$bin" "$prefix"/lib*/libdmumps.so; do
    deps=$(ldd "$lib")
    [[ $deps != *"not found"* ]] || stack_die "palace smoke: ${lib##*/} has missing libraries: $deps"
    [[ $deps == *"$blas_line"* ]] ||
      stack_die "palace smoke: ${lib##*/} does not load $blas_lib: $deps"
    case $STACK_BLAS_VENDOR in
    openblas)
      ! grep -qE '^\s*(libopenblas\.so|libmkl|libiomp5)' <<<"$deps" ||
        stack_die "palace smoke: ${lib##*/} loads another BLAS or OpenMP runtime: $deps"
      ;;
    mkl)
      ! grep -qE '^\s*(libopenblas|libiomp5|libmkl_(scalapack|blacs|intel_thread|core|gf|intel_lp))' <<<"$deps" ||
        stack_die "palace smoke: ${lib##*/} loads another BLAS, MKL ScaLAPACK or OpenMP runtime: $deps"
      ;;
    esac
  done
  if [[ $STACK_BLAS_VENDOR == mkl ]]; then
    deps=$(ldd "$prefix"/lib*/libdmumps.so)
    [[ $deps == *"libscalapack.so"*" => $prefix/lib"* ]] ||
      stack_die "palace smoke: MUMPS does not link the reference ScaLAPACK of $prefix: $deps"
  fi
  srun_mpi=$(palace_srun_mpi)
  out=$(OMP_NUM_THREADS=1 srun --ntasks=1 --cpus-per-task=1 "$srun_mpi" "$bin" --version 2>&1) ||
    stack_die "palace smoke: $bin --version failed: $out"
  [[ $out == *"$version"* ]] || stack_die "palace smoke: $bin --version does not report $version: $out"
  echo "palace smoke: $out"
  stack_meta smoke "command=srun --ntasks=1 --cpus-per-task=1 $srun_mpi bin/${bin##*/} --version" \
    "output=$(head -n 1 <<<"$out")" "versions:=$(stack_json_array "${out}")" \
    "blas=$(ldd "$bin" | grep -F "${blas_line%% =>*}" | xargs)" \
    "scalapack=$(ldd "$prefix"/lib*/libdmumps.so | grep -E 'scalapack' | xargs)" \
    "srun_mpi=$srun_mpi" "gemmt=none"
}
