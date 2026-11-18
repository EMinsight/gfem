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
# The release tarball; 0.3.34 is pinned by gfem's PETSc contract
# (tools/petsc_wheelhouse.sh TARBALLS), any other version is trusted on first
# fetch.
fetch_openblas() {
  local pin=
  case $1 in
  0.3.34) pin=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed ;;
  esac
  stack_tarball tarballs "OpenBLAS-$1.tar.gz" \
    "https://github.com/xianyi/OpenBLAS/archive/v$1.tar.gz" $pin
}

# The make options shared by both threading builds (ADR 0205): SPR kernels
# only, up to 256 threads, no thread pinning (Slurm and the caller pin).
OPENBLAS_OPTIONS=(TARGET=SAPPHIRERAPIDS DYNAMIC_ARCH=0 NUM_THREADS=256 NO_AFFINITY=1)
# Threading builds, as "library|make options": pthreads libopenblas for gfem's
# NGSolve and PETSc, OpenMP libopenblas_omp for Palace. The pthreads build is
# installed last, so the shared headers, openblas.pc and cmake/openblas are its.
# OpenBLAS appends LIBNAMESUFFIX as is: `_omp` gives libopenblas_omp.so.
OPENBLAS_BUILDS=(
  "openblas_omp|USE_OPENMP=1 LIBNAMESUFFIX=_omp"
  "openblas|USE_THREAD=1 USE_OPENMP=0"
)

# build_openblas VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
# Each threading build unpacks its own tree under BUILD_DIR ($TMPDIR). The
# toolchain flags reach OpenBLAS through COMMON_OPT/FCOMMON_OPT in the
# environment: on make's command line they would drop OpenBLAS's own
# additions (gfortran's -frecursive), and without them its -O2 comes last.
build_openblas() {
  local version=$1 prefix=$2 dir=$3 tarball entry lib opts tree builds=()
  tarball=$STACK_SRC/tarballs/OpenBLAS-$version.tar.gz
  [[ -f $tarball ]] || stack_die "no $tarball: run stack.sh fetch openblas $version"
  for entry in "${OPENBLAS_BUILDS[@]}"; do
    lib=${entry%%|*}
    read -ra opts <<<"${entry#*|}"
    opts=("${OPENBLAS_OPTIONS[@]}" "${opts[@]}" "CC=$CC" "FC=$FC")
    tree=$dir/$lib
    mkdir -p "$tree"
    tar -xzf "$tarball" -C "$tree" --strip-components=1
    echo "openblas: building lib$lib.so: make ${opts[*]}"
    COMMON_OPT=$CFLAGS FCOMMON_OPT=$FFLAGS make -C "$tree" -j "$STACK_JOBS" "${opts[@]}"
    COMMON_OPT=$CFLAGS FCOMMON_OPT=$FFLAGS make -C "$tree" "${opts[@]}" "PREFIX=$prefix" install
    if [[ $lib != openblas ]]; then
      # Keep this build's pkg-config file; the pthreads install overwrites openblas.pc.
      mv "$prefix/lib/pkgconfig/openblas.pc" "$prefix/lib/pkgconfig/$lib.pc"
    fi
    builds+=("$lib:=$(stack_json - "library=lib$lib.so" "make=${opts[*]}")")
  done
  stack_meta source "url=https://github.com/xianyi/OpenBLAS/archive/v$version.tar.gz" "tag=v$version" \
    "tarball=$(basename "$tarball")" "sha256=$(sha256sum "$tarball" | cut -d' ' -f1)"
  stack_meta configure "common_opt=$CFLAGS" "fcommon_opt=$FFLAGS" "ldflags=$LDFLAGS" \
    "builds:=$(stack_json - "${builds[@]}")"
}

# smoke_openblas VERSION PREFIX: checks the install; records it with stack_meta smoke.
# Per library, a gfortran program linked against it: a threaded dgemm and a
# zdotc (gfortran's complex-return convention) against exact values, and
# OpenBLAS's own report of its build (version, core, MAX_THREADS, threading).
smoke_openblas() {
  local version=$1 prefix=$2 dir entry lib want wantomp omp gomp out dyn needed token libs=()
  dir=$(mktemp -d "$TMPDIR/openblas-smoke.XXXXXX")
  openblas_smoke_source >"$dir/smoke.f90"
  for entry in "${OPENBLAS_BUILDS[@]}"; do
    lib=${entry%%|*}
    if [[ $entry == *USE_OPENMP=1* ]]; then want=2 wantomp=yes; else want=1 wantomp=no; fi
    # shellcheck disable=SC2086 # LDFLAGS holds several flags
    $FC -O2 "$dir/smoke.f90" -o "$dir/smoke-$lib" -L"$prefix/lib" -l"$lib" \
      -Wl,-rpath,"$prefix/lib" $LDFLAGS ||
      stack_die "openblas smoke: cannot link a gfortran program against lib$lib.so"
    out=$("$dir/smoke-$lib" "$want" 2>&1) ||
      stack_die "openblas smoke: lib$lib.so failed: $out"
    for token in "OpenBLAS $version " " SAPPHIRERAPIDS " " NO_AFFINITY " " MAX_THREADS=256"; do
      [[ "$out " == *"$token"* ]] || stack_die "openblas smoke: lib$lib.so reports no '$token': $out"
    done
    [[ $out != *DYNAMIC_ARCH* ]] || stack_die "openblas smoke: lib$lib.so is a DYNAMIC_ARCH build: $out"
    # OpenMP iff the OpenMP build: in OpenBLAS's config and as libgomp.
    dyn=$(readelf -d "$prefix/lib/lib$lib.so")
    omp=no gomp=no
    [[ "$out " != *" USE_OPENMP "* ]] || omp=yes
    [[ $dyn != *"[libgomp.so"* ]] || gomp=yes
    [[ $omp == "$wantomp" && $gomp == "$wantomp" ]] ||
      stack_die "openblas smoke: lib$lib.so: USE_OPENMP $omp, libgomp $gomp, want $wantomp: $out"
    # GCC 12.3's runtime by rpath (toolchain LDFLAGS).
    [[ $dyn == *"$STACK_GCC_LIBDIR"* ]] ||
      stack_die "openblas smoke: lib$lib.so has no rpath to $STACK_GCC_LIBDIR"
    needed=$(sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' <<<"$dyn" | paste -sd' ')
    echo "openblas smoke: lib$lib.so: $out; needs $needed"
    libs+=("$lib=$out; needs $needed")
  done
  stack_json "$dir/libraries.json" "${libs[@]}"
  stack_meta smoke "command=gfortran smoke.f90 -lLIB; ./smoke PARALLEL (dgemm n=512, zdotc; per library)" \
    "output=${libs[0]#*=}" "libraries:=$(<"$dir/libraries.json")"
  rm -rf "$dir"
}

# The smoke program. Argument: the expected openblas_get_parallel() (1
# pthreads, 2 OpenMP). Prints one line; exits non-zero on any mismatch.
openblas_smoke_source() {
  cat <<'F90'
program openblas_smoke
  use iso_c_binding
  implicit none
  interface
    integer(c_int) function openblas_get_parallel() bind(c)
      import :: c_int
    end function
    type(c_ptr) function openblas_get_config() bind(c)
      import :: c_ptr
    end function
    integer(c_int) function openblas_get_num_threads() bind(c)
      import :: c_int
    end function
  end interface
  integer, parameter :: n = 512
  double precision, allocatable :: a(:, :), b(:, :), c(:, :)
  complex(kind(0d0)) :: x(n), y(n), z, want_z
  complex(kind(0d0)), external :: zdotc
  character(len=1024), pointer :: config
  character(len=8) :: arg
  integer :: i, k, want, parallel
  call get_command_argument(1, arg)
  read (arg, *) want
  ! dgemm: C = A B with A(i, j) = i and B = 1, so C(i, j) = i n exactly.
  allocate (a(n, n), b(n, n), c(n, n))
  do i = 1, n
    a(i, :) = dble(i)
  end do
  b = 1d0
  c = -1d0
  call dgemm('N', 'N', n, n, n, 1d0, a, n, b, n, 0d0, c, n)
  do i = 1, n
    if (any(c(i, :) /= dble(i) * n)) error stop 'dgemm wrong'
  end do
  ! zdotc: x(k) = k + i, y(k) = 1 + i k, so sum conj(x) y = n(n+1) + i(sum k^2 - n).
  do k = 1, n
    x(k) = cmplx(dble(k), 1d0, kind(0d0))
    y(k) = cmplx(1d0, dble(k), kind(0d0))
  end do
  want_z = cmplx(dble(n) * (n + 1), dble(n) * (n + 1) * (2 * n + 1) / 6 - n, kind(0d0))
  z = zdotc(n, x, 1, y, 1)
  if (z /= want_z) error stop 'zdotc wrong'
  parallel = openblas_get_parallel()
  if (parallel /= want) error stop 'wrong threading layer'
  call c_f_pointer(openblas_get_config(), config)
  write (*, '(a, i0, a, i0, a, i0, 2a)') 'dgemm ok (n=', n, '), zdotc ok, parallel=', parallel, &
    ', threads=', openblas_get_num_threads(), ', config: ', config(1:index(config, c_null_char) - 1)
end program
F90
}
