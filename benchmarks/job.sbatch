#!/bin/bash
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --job-name=bench
# One exclusive Slurm job of a planned run directory: job-level facts once, then every
# line of jobs/<JOB>.steps in order, each its own `python -m benchmarks step` (port of
# tools/scale_job.sbatch). `python -m benchmarks submit` writes the command line:
#
#   sbatch --parsable --account=.. --qos=.. --time=.. --nodes=N --exclusive --ear=off \
#       [--constraint=..] --output=RUNDIR/logs/job-%j.out \
#       --export=ALL,BENCH_TREE=<shipped tree>,BENCH_PYTHON=<python> \
#       benchmarks/job.sbatch RUNDIR JOB
#
# Each step writes a STARTED stub before its run (an earlier record of the name moves
# aside to NAME.superseded-N.json) and closes it after, so a job killed here still
# leaves a row; `python -m benchmarks close RUNDIR` turns such stubs into `killed`.
# A step exiting 3 means Slurm disabled step creation: the job stops, and `close`
# stamps the lines it never reached `not_started`.
set -u
RUNDIR=${1:?usage: job.sbatch RUNDIR JOB}
JOB=${2:?usage: job.sbatch RUNDIR JOB}
: "${BENCH_TREE:?set by submit: the shipped tree}"
: "${BENCH_PYTHON:?set by submit: a Python 3.12 the compute nodes see}"
cd "$BENCH_TREE" || exit 2
module purge >/dev/null 2>&1
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NGS_NUM_THREADS
unset OMP_PLACES OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY
JOBINFO=$(scontrol show job "$SLURM_JOB_ID" -o)
BENCH_QOS=$(sed -n 's/.* QOS=\([^ ]*\).*/\1/p' <<<"$JOBINFO")
BENCH_CONSTRAINT=$(sed -n 's/.* Features=\([^ ]*\).*/\1/p' <<<"$JOBINFO")
[ "$BENCH_CONSTRAINT" = "(null)" ] && BENCH_CONSTRAINT=
LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>=3)s+=3600*$(n-2); if(n>=4)s+=86400*$(n-3); print s}')
BENCH_JOB_END=$(( $(date +%s) + LEFT ))
export BENCH_QOS BENCH_CONSTRAINT BENCH_JOB_END
STEPS=$RUNDIR/jobs/$JOB.steps
N=$(grep -c . "$STEPS")
echo "job $SLURM_JOB_ID nodes $(scontrol show hostnames | tr '\n' ' ')qos $BENCH_QOS constraint ${BENCH_CONSTRAINT:-none} job $JOB lines $N commit $(cat COMMIT 2>/dev/null) $(date +%T)"
"$BENCH_PYTHON" -m benchmarks.harness.slurm start "$RUNDIR" "$JOB" || echo "job facts incomplete (rc=$?)"
for ((i = 1; i <= N; i++)); do
  echo "--- line $i/$N $(date +%T)"
  "$BENCH_PYTHON" -m benchmarks step "$RUNDIR" "$JOB" "$i"
  rc=$?
  if [ "$rc" = 3 ]; then echo "STOP: step creation disabled"; exit 3; fi
  [ "$rc" != 0 ] && echo "step $i exited $rc"
done
echo "done $(date +%T)"
