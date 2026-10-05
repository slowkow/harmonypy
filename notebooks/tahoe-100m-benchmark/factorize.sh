#!/bin/bash
# Compare main with venv-factorize, a copy whose run_harmony encodes the batch
# labels with pd.factorize(..., sort=True) instead of np.unique (same codes,
# so the same results). The two alternate, three repeats per dataset, into
# results-factorize.jsonl. Waits for run.sh to release the lock first.
set -u
cd "$(dirname "$0")"
exec 9> bench.lock
flock 9
export BENCH_RUN_ID=factorize-$(date +%Y%m%dT%H%M%S)
echo "== $(date '+%F %T') start"
for d in 1M 16M full; do
    for rep in 1 2 3; do
        for v in main factorize; do
            echo "== $(date '+%F %T') $v $d"
            venv-$v/bin/python bench.py --build $v --dataset $d --sweep factorize \
                --results results-factorize.jsonl > /dev/null 2>> errors.log || echo "FAILED $v $d"
        done
    done
done
echo "== $(date '+%F %T') done"
