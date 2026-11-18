#!/usr/bin/env bash
# Build the native software stacks (native-oss, native-intel) of the native
# builds effort on a Slurm cluster: Palace, NGSolve, PETSc/SLEPc, OpenBLAS and
# gfem's stack venv, each into its own prefix
#   $STACKS_ROOT/<stack>/<pkg>-<version>/
# with a provenance file (provenance.json) written last, which marks it
# complete. Developer tool: see README.md beside this script.
#
# Usage:
#   stack.sh fetch  PKG [VERSION]
#       login node, network only: fill the source mirror $STACKS_ROOT/src
#   stack.sh submit [--rebuild] [--allow-dirty] STACK PKG [VERSION] [-- SBATCH_ARGS...]
#       login node: sbatch build.sbatch for one package (job stack-<stack>-<pkg>-<version>)
#   stack.sh build  [--rebuild] [--allow-dirty] STACK PKG [VERSION]
#       inside the Slurm job only, offline: build, smoke, write provenance.json
#   stack.sh check  [STACK [PKG [VERSION]]]
#       read-only: state of the prefixes; fails when one stack's complete
#       prefixes name different toolchain hashes or their MUMPS blocks differ
#       in a must-match field
#
# STACKS_ROOT (required) is the install root. VERSION defaults to the pin in
# packages/<pkg>.sh. STACK_PYTHON names the Python that runs provenance.py
# (default python3; 3.9 or newer). STACK_QOS is the default QOS of submit
# (default gp_ehpc). <DEP>_VERSION (e.g. OPENBLAS_VERSION) picks the version
# of a dependency other than its recipe's default.
set -euo pipefail

STACK_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export STACK_HOME
STACK_PYTHON=${STACK_PYTHON:-python3}
# Directories of $STACKS_ROOT that are not software stacks.
RESERVED_DIRS=(src recipes logs)
PACKAGES=(openblas ngsolve petsc palace pyenv)

die() {
  echo "stack.sh: $*" >&2
  exit 1
}

stack_log() {
  echo "stack.sh: $*"
}

prov() {
  "$STACK_PYTHON" -B "$STACK_HOME/provenance.py" "$@"
}

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//' >&2
  exit 2
}

# --- helpers for recipes (see README.md, "Recipe interface") ----------------

stack_die() {
  die "$@"
}

# stack_json FILE KEY=STRING KEY:=JSON ...: write one JSON object to FILE
# (standard output for -). Build nested values with a nested stack_json -,
# never by pasting strings into JSON.
stack_json() {
  local file=$1
  shift
  "$STACK_PYTHON" -B - "$file" "$@" <<'PY'
import json
import sys

out = {}
for arg in sys.argv[2:]:
    if arg.partition("=")[0].endswith(":"):
        key, _, value = arg.partition(":=")
        out[key] = json.loads(value)
    else:
        key, value = arg.split("=", 1)
        out[key] = value
text = json.dumps(out, indent=2, sort_keys=True) + "\n"
if sys.argv[1] == "-":
    sys.stdout.write(text)
else:
    with open(sys.argv[1], "w") as handle:
        handle.write(text)
PY
}

# stack_json_array STRING...: print the strings as one JSON array.
stack_json_array() {
  "$STACK_PYTHON" -B -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@"
}

# stack_meta FIELD KEY=STRING KEY:=JSON ...: one field of the provenance file.
stack_meta() {
  local field=$1
  shift
  [[ -n ${STACK_META:-} ]] || die "stack_meta outside a build"
  stack_json "$STACK_META/$field.json" "$@"
}

# stack_smoke COMMAND...: run the smoke command, record it and its first line.
stack_smoke() {
  local output
  output=$("$@" 2>&1) || die "smoke command failed: $*"$'\n'"$output"
  stack_meta smoke "command=$*" "output=$(head -n 1 <<<"$output")"
}

# stack_tofu FILE: sha256 of a tarball no upstream file pins, trusted on first
# fetch and checked on every later one (ADR 0212).
stack_tofu() {
  prov tofu "$STACKS_ROOT/src/tarballs/SHA256SUMS" "$1"
}

# stack_singleton COMMAND...: run COMMAND, an MPI singleton (one process
# outside srun), without I_MPI_PMI_LIBRARY: Intel MPI singletons fail MPI_Init
# with the module's Slurm libpmi2 (job 49181364, probe 49181762); mpiexec
# needs none.
stack_singleton() {
  env -u I_MPI_PMI_LIBRARY "$@"
}

# --- helpers for fetch_<pkg> (login node, network) ---------------------------
# Each reports one line, "new" or "present"; fetch counts them, so a rerun on
# a complete mirror says "nothing new".

FETCH_NEW=0
FETCH_PRESENT=0

fetch_report() {
  if [[ $1 == new ]]; then
    FETCH_NEW=$((FETCH_NEW + 1))
  else
    FETCH_PRESENT=$((FETCH_PRESENT + 1))
  fi
  echo "  $1: $2"
}

# stack_mirror_dir URL: the bare mirror of URL, at mirrors/<host>/<path>.git
# with the URL spelled as upstream spells it (job-local insteadOf maps it back).
stack_mirror_dir() {
  local rel=${1#*://}
  echo "$STACK_SRC/mirrors/${rel%.git}.git"
}

# stack_mirror URL REF...: a full bare mirror of URL (git clone --mirror) that
# holds every REF (tag, branch or commit). A mirror that already holds them is
# left alone; one that lacks one is updated (git remote update --prune).
stack_mirror() {
  local url=$1 ref missing=()
  shift
  local dst
  dst=$(stack_mirror_dir "$url")
  local rel=${dst#"$STACK_SRC"/}
  if [[ -d $dst ]]; then
    for ref in "$@"; do
      git --git-dir="$dst" rev-parse -q --verify "$ref^{commit}" >/dev/null || missing+=("$ref")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
      fetch_report present "$rel"
      return 0
    fi
    git --git-dir="$dst" remote update --prune >/dev/null ||
      die "git remote update of $rel failed"
  else
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst.part"
    git clone -q --mirror "$url" "$dst.part" || die "git clone --mirror $url failed"
    mv "$dst.part" "$dst"
  fi
  for ref in "$@"; do
    git --git-dir="$dst" rev-parse -q --verify "$ref^{commit}" >/dev/null ||
      die "$url has no $ref"
  done
  fetch_report new "$rel${missing[*]:+ (updated for ${missing[*]})}"
}

# stack_show URL REF FILE: FILE at REF in the mirror of URL.
stack_show() {
  git --git-dir="$(stack_mirror_dir "$1")" show "$2:$3"
}

# stack_sources COMMAND...: sources.py, which reads pins from a fetched tag.
stack_sources() {
  "$STACK_PYTHON" -B "$STACK_HOME/sources.py" "$@"
}

# stack_mirror_submodules URL REF: mirror each submodule of URL at REF, at the
# commit REF records, recursively (.gitmodules spelling, as git will ask).
stack_mirror_submodules() {
  local url=$1 ref=$2 dir modules line path suburl commit
  dir=$(stack_mirror_dir "$url")
  git --git-dir="$dir" cat-file -e "$ref:.gitmodules" 2>/dev/null || return 0
  modules=$(stack_show "$url" "$ref" .gitmodules | stack_sources gitmodules)
  local lines=()
  mapfile -t lines <<<"$modules"
  for line in "${lines[@]}"; do
    [[ -n $line ]] || continue
    read -r path suburl <<<"$line"
    [[ $suburl == *://* ]] || die "$url: relative submodule URL $suburl is not supported"
    commit=$(git --git-dir="$dir" rev-parse "$ref:$path")
    stack_mirror "$suburl" "$commit"
    stack_mirror_submodules "$suburl" "$commit"
  done
}

# stack_download DIR NAME URL: download URL as $STACK_SRC/DIR/NAME once, with
# no hash: only for a file that is itself the pin of others (a SHA256SUMS).
stack_download() {
  local file=$STACK_SRC/$1/$2
  if [[ -f $file ]]; then
    fetch_report present "$1/$2"
    return 0
  fi
  mkdir -p "$STACK_SRC/$1"
  curl -fsSL --retry 3 -o "$file.part" "$3" || {
    rm -f "$file.part"
    die "download of $3 failed"
  }
  mv "$file.part" "$file"
  fetch_report new "$1/$2"
}

# stack_pip wheels|nodeps|sdists REQ...: pip download NAME==VERSION requirements
# into $STACK_SRC/pip/wheels (cp312 manylinux wheels and their dependencies;
# nodeps: without them) or pip/sdists (source only, no dependencies), with the
# cluster's own pip (pip honours the tunnel; uv's module is offline). Nothing
# is downloaded when every requirement already has its file.
stack_pip() {
  local kind=$1 req name version file have missing=()
  shift
  local dir=$STACK_SRC/pip/wheels args=()
  case $kind in
  wheels | nodeps)
    args=(--only-binary=:all: --python-version 3.12 --implementation cp
      --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64
      --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64
      --platform manylinux_2_5_x86_64 --platform manylinux1_x86_64 --platform any)
    [[ $kind == wheels ]] || args+=(--no-deps)
    ;;
  sdists)
    dir=$STACK_SRC/pip/sdists
    args=(--no-binary=:all: --no-deps)
    ;;
  *) die "stack_pip: unknown kind $kind" ;;
  esac
  mkdir -p "$dir"
  for req in "$@"; do
    name=${req%%==*} version=${req#*==}
    [[ $req == *==* && -n $version ]] || die "stack_pip: $req is not NAME==VERSION"
    name=$(tr 'A-Z.-' 'a-z__' <<<"$name")
    have=false
    for file in "$dir"/*; do
      file=$(tr 'A-Z.-' 'a-z__' <<<"${file##*/}")
      if [[ $file == "${name}_$(tr '.-' '__' <<<"$version")_"* ]]; then
        have=true
        break
      fi
    done
    $have || missing+=("$req")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    fetch_report present "pip/${dir##*/}: $*"
    return 0
  fi
  local before after
  before=$(ls "$dir")
  "$STACK_PYTHON" -m pip download -q --disable-pip-version-check -d "$dir" "${args[@]}" "$@" ||
    die "pip download of $* failed"
  after=$(ls "$dir")
  fetch_report new "pip/${dir##*/}: $(comm -13 <(echo "$before") <(echo "$after") | tr '\n' ' ')"
}

# stack_tarball DIR NAME URL [SHA256]: download URL as $STACK_SRC/DIR/NAME. With
# SHA256 (a pin from an upstream file) the file must match it; without, its
# hash is trusted on first fetch and checked after (stack_tofu). A file that
# fails its check is refused; a new one is not kept.
stack_tarball() {
  local dir=$STACK_SRC/$1 name=$2 url=$3 pin=${4:-}
  local file=$dir/$name state=present
  mkdir -p "$dir"
  if [[ ! -f $file ]]; then
    rm -f "$file.part"
    curl -fsSL --retry 3 -o "$file.part" "$url" || {
      rm -f "$file.part"
      die "download of $url failed"
    }
    mv "$file.part" "$file"
    state=new
  fi
  if [[ -n $pin ]]; then
    local got
    got=$(prov sha256 "$file")
    if [[ $got != "$pin" ]]; then
      [[ $state == present ]] || rm -f "$file"
      die "$name: sha256 $got differs from the pin $pin ($url)"
    fi
    fetch_report "$state" "$1/$name (sha256 pinned)"
  else
    local outcome
    outcome=$(stack_tofu "$file") || {
      [[ $state == present ]] || rm -f "$file"
      die "$name: refused (sha256 differs from the one recorded on first fetch)"
    }
    [[ $outcome != *recorded ]] || state=new
    fetch_report "$state" "$1/$name (sha256 ${outcome##* })"
  fi
}

# --- recipes and stacks ------------------------------------------------------

require_root() {
  [[ -n ${STACKS_ROOT:-} ]] ||
    die "STACKS_ROOT is not set: export the install root of the software stacks"
  STACKS_ROOT=${STACKS_ROOT%/}
  export STACKS_ROOT
}

require_stack() {
  [[ -f $STACK_HOME/toolchains/$1.sh ]] ||
    die "unknown software stack '$1': no toolchains/$1.sh"
}

# load_recipe PKG: source packages/PKG.sh, with STACK_NAME set (empty for fetch).
load_recipe() {
  local pkg=$1
  [[ -f $STACK_HOME/packages/$pkg.sh ]] || die "unknown package '$pkg': no packages/$pkg.sh"
  unset VERSION CPUS TIME
  DEPENDS=()
  STACKS=()
  # shellcheck source=/dev/null
  source "$STACK_HOME/packages/$pkg.sh"
  [[ -n ${VERSION:-} ]] || die "packages/$pkg.sh sets no VERSION"
  [[ -n ${CPUS:-} && -n ${TIME:-} ]] ||
    die "packages/$pkg.sh sets no CPUS or TIME"
  local fn
  for fn in "fetch_$pkg" "build_$pkg" "smoke_$pkg"; do
    declare -F "$fn" >/dev/null || die "packages/$pkg.sh defines no $fn"
  done
}

builds_on() {
  local stack=$1 s
  for s in "${STACKS[@]}"; do
    [[ $s == "$stack" ]] && return 0
  done
  return 1
}

# dep_version DEP: <DEP>_VERSION from the environment, else DEP's default.
dep_version() {
  local dep=$1 var
  var="$(tr '[:lower:]-' '[:upper:]_' <<<"$dep")_VERSION"
  if [[ -n ${!var:-} ]]; then
    echo "${!var}"
  else
    (load_recipe "$dep" && echo "$VERSION")
  fi
}

toolchain_sha() {
  local stack=$1 frozen=$STACKS_ROOT/$1/toolchain.sh
  if [[ -f $frozen ]]; then
    prov sha256 "$frozen"
  else
    prov sha256 "$STACK_HOME/toolchains/$stack.sh"
  fi
}

# recipe_commit: sets RECIPE_COMMIT and RECIPE_DIRTY (true, false or unknown).
# A git checkout names its HEAD and may be dirty; a tree shipped by the
# harness (git archive, no .git) names its commit in $STACKS_RECIPE_COMMIT or
# a COMMIT file and is clean by construction.
recipe_commit() {
  if [[ $(git -C "$STACK_HOME" rev-parse --is-inside-work-tree 2>/dev/null) == true ]]; then
    RECIPE_COMMIT=$(git -C "$STACK_HOME" rev-parse HEAD)
    if [[ -n $(git -C "$STACK_HOME" status --porcelain -- .) ]]; then
      RECIPE_DIRTY=true
    else
      RECIPE_DIRTY=false
    fi
  elif [[ -n ${STACKS_RECIPE_COMMIT:-} ]]; then
    RECIPE_COMMIT=$STACKS_RECIPE_COMMIT
    RECIPE_DIRTY=false
  elif [[ -s $STACK_HOME/COMMIT ]]; then
    RECIPE_COMMIT=$(head -n 1 "$STACK_HOME/COMMIT")
    RECIPE_DIRTY=false
  else
    RECIPE_COMMIT=unknown
    RECIPE_DIRTY=unknown
  fi
}

source_quietly() {
  # Module scripts read unset variables; the toolchain and activate files run
  # without -u.
  set +u
  # shellcheck source=/dev/null
  source "$1"
  set -u
}

# stack_freeze_toolchain TOOLCHAIN FROZEN: freeze TOOLCHAIN as FROZEN at a
# stack's first build, then refuse any build whose TOOLCHAIN differs. Two
# first builds may race: each writes a private read-only copy beside FROZEN
# and publishes it with ln, which never replaces an existing file, so FROZEN
# appears whole or not at all and the loser compares against the winner's.
stack_freeze_toolchain() {
  local toolchain=$1 frozen=$2 tmp
  if [[ ! -f $frozen ]]; then
    tmp=$frozen.tmp.$(hostname).$BASHPID
    rm -f "$tmp"
    cp "$toolchain" "$tmp"
    chmod a-w "$tmp"
    ln "$tmp" "$frozen" 2>/dev/null || [[ -f $frozen ]] ||
      die "cannot freeze $toolchain as $frozen"
    rm -f "$tmp"
  fi
  cmp -s "$frozen" "$toolchain" ||
    die "$toolchain differs from $frozen, frozen at the stack's first build: a changed toolchain is a new software stack"
}

# stack_jobs NAME: the ids of this user's pending or running jobs named NAME,
# one per line. A failed squeue fails the caller: it never reads as "no job".
stack_jobs() {
  squeue --me --noheader --name="$1" --format=%i ||
    die "squeue failed: cannot tell whether a job named $1 is pending or running"
}

# --- commands ----------------------------------------------------------------

cmd_fetch() {
  [[ $# -ge 1 && $# -le 2 ]] || usage
  local pkg=$1
  [[ -z ${SLURM_JOB_ID:-} ]] || die "fetch runs on the login node (network), not in a job"
  STACK_NAME=
  load_recipe "$pkg"
  local version=${2:-$VERSION}
  export STACK_SRC=$STACKS_ROOT/src
  mkdir -p "$STACK_SRC"/{mirrors,tarballs,patches,pip,python}
  stack_log "fetch $pkg $version into $STACK_SRC"
  "fetch_$pkg" "$version"
  if [[ $FETCH_NEW -eq 0 ]]; then
    stack_log "fetch $pkg $version: nothing new ($FETCH_PRESENT present)"
  else
    stack_log "fetch $pkg $version: $FETCH_NEW new, $FETCH_PRESENT present"
  fi
}

cmd_submit() {
  local flags=() extra=() args=()
  while [[ $# -gt 0 ]]; do
    case $1 in
    --rebuild | --allow-dirty) flags+=("$1") ;;
    --)
      shift
      extra=("$@")
      break
      ;;
    *) args+=("$1") ;;
    esac
    shift
  done
  [[ ${#args[@]} -ge 2 && ${#args[@]} -le 3 ]] || usage
  local stack=${args[0]} pkg=${args[1]}
  require_stack "$stack"
  STACK_NAME=$stack
  load_recipe "$pkg"
  builds_on "$stack" || die "$pkg is not built on $stack (packages/$pkg.sh STACKS)"
  local version=${args[2]:-$VERSION}
  local name="stack-$stack-$pkg-$version"
  local prefix=$STACKS_ROOT/$stack/$pkg-$version
  local deps=("${DEPENDS[@]}") cpus=$CPUS time=$TIME

  if [[ " ${flags[*]} " != *" --rebuild "* ]]; then
    local now
    now=$(prov state "$prefix" "$(prov recipe-sha256 "$STACK_HOME" "$pkg")" "$(toolchain_sha "$stack")")
    [[ $now != complete ]] || {
      stack_log "$stack/$pkg-$version is complete with this recipe and toolchain: nothing to submit"
      return 0
    }
    [[ $now != differs ]] ||
      die "$stack/$pkg-$version was built from another recipe or toolchain: pass --rebuild to replace it"
  fi
  local jobs
  jobs=$(stack_jobs "$name")
  [[ -z $jobs ]] || die "a job named $name is already pending or running"

  local after=() dep depver id
  for dep in "${deps[@]}"; do
    depver=$(dep_version "$dep")
    [[ -f $STACKS_ROOT/$stack/$dep-$depver/provenance.json ]] && continue
    jobs=$(stack_jobs "stack-$stack-$dep-$depver")
    id=$(head -n 1 <<<"$jobs")
    [[ -n $id ]] || die "$stack/$dep-$depver is not built and no job builds it: submit $dep first"
    after+=("$id")
  done
  local dependency=()
  if [[ ${#after[@]} -gt 0 ]]; then
    dependency=("--dependency=afterok:$(
      IFS=:
      echo "${after[*]}"
    )")
  fi

  mkdir -p "$STACKS_ROOT/logs"
  sbatch \
    --job-name="$name" \
    --nodes=1 --ntasks=1 \
    --cpus-per-task="$cpus" \
    --hint=nomultithread \
    --time="$time" \
    --qos="${STACK_QOS:-gp_ehpc}" \
    --output="$STACKS_ROOT/logs/$stack-$pkg-$version-%j.out" \
    --export="ALL,STACK_HOME=$STACK_HOME" \
    "${dependency[@]}" "${extra[@]}" \
    "$STACK_HOME/build.sbatch" "${flags[@]}" "$stack" "$pkg" "$version"
}

on_build_exit() {
  local status=$?
  if [[ $status -ne 0 && -n ${FAILED_DIR:-} && -d ${BUILD_DIR:-} ]]; then
    mkdir -p "$FAILED_DIR"
    find "$BUILD_DIR" -maxdepth 3 \( -name '*.log' -o -name CMakeCache.txt -o -name 'CMake*.log' \) \
      -exec cp --parents -t "$FAILED_DIR" {} + 2>/dev/null || true
    [[ -d $BUILD_DIR/meta ]] && cp -r "$BUILD_DIR/meta" "$FAILED_DIR/" || true
    echo "stack.sh: build failed; logs in $FAILED_DIR, prefix left incomplete" >&2
  fi
  return "$status"
}

cmd_build() {
  local rebuild=false allow_dirty=false args=()
  while [[ $# -gt 0 ]]; do
    case $1 in
    --rebuild) rebuild=true ;;
    --allow-dirty) allow_dirty=true ;;
    *) args+=("$1") ;;
    esac
    shift
  done
  [[ ${#args[@]} -ge 2 && ${#args[@]} -le 3 ]] || usage
  [[ -n ${SLURM_JOB_ID:-} ]] ||
    die "build runs inside its Slurm job only (no SLURM_JOB_ID): use submit"
  local stack=${args[0]} pkg=${args[1]}
  require_stack "$stack"
  STACK_NAME=$stack
  export STACK_NAME
  load_recipe "$pkg"
  builds_on "$stack" || die "$pkg is not built on $stack (packages/$pkg.sh STACKS)"
  local version=${args[2]:-$VERSION}
  local deps=("${DEPENDS[@]}")
  [[ -n ${TMPDIR:-} && $TMPDIR != /tmp ]] ||
    die "TMPDIR must name the job's node-local scratch, not /tmp or unset"

  recipe_commit
  if [[ $RECIPE_DIRTY != false && $allow_dirty != true ]]; then
    die "the recipe tree is dirty or of unknown commit ($RECIPE_COMMIT): commit it or pass --allow-dirty"
  fi

  local stack_dir=$STACKS_ROOT/$stack
  local prefix=$stack_dir/$pkg-$version
  local frozen=$stack_dir/toolchain.sh
  mkdir -p "$stack_dir"
  stack_freeze_toolchain "$STACK_HOME/toolchains/$stack.sh" "$frozen"
  local recipe_sha toolchain_sha now
  recipe_sha=$(prov recipe-sha256 "$STACK_HOME" "$pkg")
  toolchain_sha=$(prov sha256 "$frozen")
  now=$(prov state "$prefix" "$recipe_sha" "$toolchain_sha")
  case $now in
  complete)
    stack_log "$stack/$pkg-$version is complete with this recipe and toolchain: nothing to do"
    return 0
    ;;
  differs)
    $rebuild || die "$stack/$pkg-$version was built from another recipe or toolchain: pass --rebuild to replace it"
    mkdir -p "$stack_dir/.history"
    mv "$prefix/provenance.json" "$stack_dir/.history/$pkg-$version-$(date -u +%Y%m%dT%H%M%SZ).json"
    rm -rf "$prefix"
    ;;
  incomplete)
    stack_log "wiping the leftovers of an unfinished build in $prefix"
    rm -rf "$prefix"
    ;;
  esac

  local start
  start=$(date +%s)
  source_quietly "$frozen"
  export STACK_TOOLCHAIN=$frozen

  local dep depver depprefix depends_json=() activates=()
  for dep in "${deps[@]}"; do
    depver=$(dep_version "$dep")
    depprefix=$stack_dir/$dep-$depver
    [[ -f $depprefix/provenance.json ]] ||
      die "dependency $stack/$dep-$depver is not complete (no provenance.json)"
    source_quietly "$depprefix/activate.sh"
    activates+=("$depprefix/activate.sh")
    depends_json+=("$dep:=$(stack_json - "prefix=$depprefix" \
      "provenance_sha256=$(prov sha256 "$depprefix/provenance.json")")")
  done

  BUILD_DIR=$TMPDIR/stack-$stack-$pkg-$version
  FAILED_DIR=$stack_dir/.failed/$pkg-$version-$SLURM_JOB_ID
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR/meta" "$prefix"
  export STACK_META=$BUILD_DIR/meta STACK_SRC=$STACKS_ROOT/src STACK_JOBS=${SLURM_CPUS_PER_TASK:-1}
  trap on_build_exit EXIT

  "build_$pkg" "$version" "$prefix" "$BUILD_DIR" 2>&1 | tee "$BUILD_DIR/build.log"
  local field
  for field in source configure; do
    [[ -f $STACK_META/$field.json ]] || die "build_$pkg wrote no $field.json"
  done
  # The smoke step runs against the prefix as a consumer sees it.
  write_activate "$stack" "$pkg" "$prefix" "$frozen" "${activates[@]}"
  source_quietly "$prefix/activate.sh"
  "smoke_$pkg" "$version" "$prefix"
  [[ -f $STACK_META/smoke.json ]] || die "smoke_$pkg recorded no smoke (use stack_smoke)"

  mkdir -p "$prefix/share/stack"
  cp "$BUILD_DIR/build.log" "$prefix/share/stack/"
  tar -C "$STACK_HOME" --exclude=__pycache__ -cf "$prefix/share/stack/recipe.tar" .

  module -t list >"$BUILD_DIR/modules.txt" 2>&1 || true
  prov toolchain "$frozen" "$BUILD_DIR/modules.txt" >"$STACK_META/toolchain.json"
  local dirty="dirty:=$RECIPE_DIRTY"
  [[ $RECIPE_DIRTY != unknown ]] || dirty="dirty=unknown"
  stack_meta recipe "commit=$RECIPE_COMMIT" "sha256=$recipe_sha" "$dirty"
  stack_meta depends "${depends_json[@]}"
  local end cputime
  end=$(date +%s)
  cputime=$(sacct -j "$SLURM_JOB_ID" -X -n -P -o CPUTimeRAW 2>/dev/null | head -n 1 || true)
  stack_meta build "job_id=$SLURM_JOB_ID" "host=$(hostname)" "qos=${SLURM_JOB_QOS:-}" \
    "cpus:=${SLURM_CPUS_PER_TASK:-1}" "start_utc=$(date -u -d "@$start" +%Y-%m-%dT%H:%M:%SZ)" \
    "end_utc=$(date -u -d "@$end" +%Y-%m-%dT%H:%M:%SZ)" "elapsed_s:=$((end - start))" \
    "cputime_raw_so_far:=${cputime:-null}"
  echo "\"$pkg\"" >"$STACK_META/package.json"
  echo "\"$version\"" >"$STACK_META/version.json"
  echo "\"$stack\"" >"$STACK_META/stack.json"
  prov write "$prefix" "$STACK_META"
  stack_log "$stack/$pkg-$version complete"
}

# write_activate STACK PKG PREFIX TOOLCHAIN [DEP_ACTIVATE...]
write_activate() {
  local stack=$1 pkg=$2 prefix=$3 toolchain=$4
  shift 4
  local upper dir file=$prefix/activate.sh
  upper=$(tr '[:lower:]-' '[:upper:]_' <<<"$pkg")
  {
    echo "# Use $stack/$(basename "$prefix"): source this file in a fresh shell."
    echo "# Written by stack.sh build (job $SLURM_JOB_ID). Sets no thread counts."
    echo "if [ \"\${STACK_TOOLCHAIN:-}\" != \"$toolchain\" ]; then"
    echo "  . \"$toolchain\""
    echo "  export STACK_TOOLCHAIN=\"$toolchain\""
    echo "fi"
    for dep in "$@"; do
      echo ". \"$dep\""
    done
    echo "export PATH=\"$prefix/bin\${PATH:+:\$PATH}\""
    for dir in lib lib64; do
      [[ -d $prefix/$dir ]] &&
        echo "export LD_LIBRARY_PATH=\"$prefix/$dir\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
      [[ -d $prefix/$dir/pkgconfig ]] &&
        echo "export PKG_CONFIG_PATH=\"$prefix/$dir/pkgconfig\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}\""
    done
    echo "export CMAKE_PREFIX_PATH=\"$prefix\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}\""
    echo "export STACK_NAME=\"$stack\""
    echo "export STACK_${upper}_PROVENANCE=\"$prefix/provenance.json\""
    if declare -F "activate_$pkg" >/dev/null; then
      "activate_$pkg" "$prefix"
    fi
  } >"$file"
}

# check_petsc_contract: packages/petsc.sh and openblas.sh's tarball pin
# against tools/petsc_wheelhouse.sh (ADR 0209): beside the recipe tree in a
# checkout, inside it when shipped.
check_petsc_contract() {
  local wheelhouse
  for wheelhouse in "$STACK_HOME/../petsc_wheelhouse.sh" "$STACK_HOME/petsc_wheelhouse.sh"; do
    if [[ -f $wheelhouse ]]; then
      prov petsc-contract "$wheelhouse" "$STACK_HOME/packages/petsc.sh" \
        "$STACK_HOME/packages/openblas.sh"
      return
    fi
  done
  echo "petsc contract: not checked (no petsc_wheelhouse.sh beside or in $STACK_HOME)"
}

cmd_check() {
  [[ $# -le 3 ]] || usage
  local stacks=() d
  if [[ $# -eq 0 ]]; then
    if [[ -d $STACKS_ROOT ]]; then
      for d in "$STACKS_ROOT"/*/; do
        [[ -d $d ]] || continue
        d=$(basename "$d")
        [[ " ${RESERVED_DIRS[*]} " == *" $d "* ]] && continue
        stacks+=("$STACKS_ROOT/$d")
      done
    fi
    if [[ ${#stacks[@]} -eq 0 ]]; then
      stack_log "no software stack under $STACKS_ROOT"
      return 0
    fi
    prov check "${stacks[@]}"
    return
  fi

  local stack=$1 status=0
  prov check "$STACKS_ROOT/$stack" || status=$?
  check_petsc_contract || status=1
  [[ -f $STACK_HOME/toolchains/$stack.sh ]] || return "$status"
  local frozen=$STACKS_ROOT/$stack/toolchain.sh
  if [[ -f $frozen ]] && ! cmp -s "$frozen" "$STACK_HOME/toolchains/$stack.sh"; then
    echo "  toolchains/$stack.sh differs from the frozen $frozen"
  fi
  STACK_NAME=$stack
  local toolchain_sha pkg version
  toolchain_sha=$(toolchain_sha "$stack")
  local pkgs=("${PACKAGES[@]}")
  [[ $# -ge 2 ]] && pkgs=("$2")
  echo "plan:"
  for pkg in "${pkgs[@]}"; do
    load_recipe "$pkg"
    builds_on "$stack" || continue
    version=$VERSION
    [[ $# -ge 3 ]] && version=$3
    echo "$stack/$pkg-$version: $(prov state "$STACKS_ROOT/$stack/$pkg-$version" \
      "$(prov recipe-sha256 "$STACK_HOME" "$pkg")" "$toolchain_sha")"
    echo "  version=$version"
    echo "  depends=${DEPENDS[*]}"
    echo "  cpus=$CPUS"
    echo "  time=$TIME"
  done
  return "$status"
}

main() {
  [[ $# -ge 1 ]] || usage
  local command=$1
  shift
  case $command in
  fetch | submit | build | check) ;;
  -h | --help | help) usage ;;
  *) die "unknown command '$command' (fetch, submit, build, check)" ;;
  esac
  require_root
  "cmd_$command" "$@"
}

# Sourced (tests of the helpers), it defines the functions and runs nothing.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
