#!/bin/bash
# Run the benchmark: one process per configuration, results appended to
# results.jsonl, progress to stdout. Within each dataset the two builds run
# back to back, so a change in the server's load affects both. A lock keeps
# two invocations from running at once.
#
# usage: ./run.sh [stage ...]
#   core       the figure: samples and cells sweeps for both builds, and the
#              threads sweeps (1M and 16M cells) for main
#   main-full  main on all 95.6M cells
#   extra      10 forced iterations (main and 2.0.2), main with 0.2's
#              defaults, and main pinned to one socket or to physical cores
#   old-full   2.0.2 on all 95.6M cells (takes hours)
# Default: core main-full extra old-full.
set -u
cd "$(dirname "$0")"
exec 9> bench.lock
flock 9
export BENCH_RUN_ID=${BENCH_RUN_ID:-$(date +%Y%m%dT%H%M%S)}
mkdir -p Z
MAIN="venv-main/bin/python bench.py --build main-623ff51"
OLD="venv-2.0.2/bin/python bench.py --build 2.0.2"

run() {
    echo "== $(date '+%F %T') $*"
    "$@" > /dev/null 2>> errors.log
    local status=$?
    if [ $status -ne 0 ]; then
        echo "FAILED (exit $status): $*"
        # bench.py records its own errors and exits 1; record crashes too,
        # for example a run killed for using too much memory.
        if [ $status -ne 1 ]; then
            venv-main/bin/python -c 'import json, os, sys; print(json.dumps({"failed": True, "exit_status": int(sys.argv[1]), "command": sys.argv[2:], "run_id": os.environ.get("BENCH_RUN_ID")}))' \
                "$status" "$@" >> results.jsonl
        fi
    fi
}

# --save the corrected coordinates of the first run on 1M and 16M cells.
save() { # dataset repeat name
    if [ "$2" = 1 ] && { [ "$1" = 1M ] || [ "$1" = 16M ]; }; then echo "--save Z/$3-$1.npy"; fi
}

for stage in ${*:-core main-full extra old-full}; do
    case $stage in
    core)
        for d in 50B 100B 200B 400B 800B; do
            for rep in 1 2 3 4 5; do run $MAIN --dataset $d --sweep samples; done
            run $OLD --dataset $d --sweep samples
        done
        for d in 1M 2M 4M 8M 16M; do
            for rep in 1 2 3 4 5; do run $MAIN --dataset $d --sweep cells $(save $d $rep main); done
            reps="1"
            if [ $d = 1M ] || [ $d = 16M ]; then reps="1 2 3"; fi
            for rep in $reps; do run $OLD --dataset $d --sweep cells $(save $d $rep 2.0.2); done
        done
        # The default (all 128 threads) is measured by the cells sweep.
        for d in 1M 16M; do
            for rep in 1 2 3; do
                for n in 1 2 4 8 16 32 64; do run $MAIN --dataset $d --sweep threads --ncores $n; done
            done
        done
        ;;
    main-full)
        for rep in 1 2 3; do run $MAIN --dataset full --sweep cells; done
        ;;
    extra)
        forced='{"max_iter_harmony": 10, "epsilon_harmony": -1}'
        for rep in 1 2 3; do
            run $MAIN --dataset 1M --sweep forced10 --kwargs "$forced"
            run $MAIN --dataset 16M --sweep forced10 --kwargs "$forced"
        done
        run $OLD --dataset 1M --sweep forced10 --kwargs "$forced"
        old_defaults='{"max_iter_kmeans": 20, "epsilon_cluster": 1e-5, "epsilon_harmony": 1e-4, "lamb": 1}'
        for rep in 1 2 3; do
            run $MAIN --dataset 1M --sweep settings-0.2 --kwargs "$old_defaults"
            run $MAIN --dataset 16M --sweep settings-0.2 --kwargs "$old_defaults"
        done
        # Socket 0 is CPUs 0-31 with hyperthreads 64-95; socket 1 is 32-63 and 96-127.
        for rep in 1 2 3; do
            run taskset -c 0-31 $MAIN --dataset 16M --sweep pinned --ncores 32
            run taskset -c 0-31,64-95 $MAIN --dataset 16M --sweep pinned --ncores 64
            run taskset -c 0-63 $MAIN --dataset 16M --sweep pinned --ncores 64
        done
        ;;
    old-full)
        run $OLD --dataset full --sweep cells
        ;;
    *)
        echo "unknown stage: $stage"
        ;;
    esac
done
echo "== $(date '+%F %T') done"
