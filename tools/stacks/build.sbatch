#!/usr/bin/env bash
# The Slurm job of one package build: `stack.sh submit` sets its name
# (stack-<stack>-<pkg>-<version>), resources, QOS and output file, and passes
# STACK_HOME, the recipe tree to run. Arguments: [--rebuild] [--allow-dirty]
# STACK PKG VERSION.
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${STACK_HOME:?build.sbatch is submitted by stack.sh submit, which sets STACK_HOME}"
exec bash "$STACK_HOME/stack.sh" build "$@"
