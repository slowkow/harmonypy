import os
import time
import pytest
import logging

from time import time
from multiprocessing import Process


def run_test_in_process(test_func, *args):
    """Wrapper to run test function in a separate process"""
    process = Process(target=test_func, args=args)
    process.start()
    process.join(timeout=30)  # 30 second timeout

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Test timed out for args: {args}")

    assert process.exitcode == 0, f"Test failed for args: {args}"


def run_it(cpu):
    os.environ['HARMONYPY_CPU'] = str(cpu)
    if cpu == 1:
        import pandas as pd
    else:
        import cudf as pd
    import harmonypy as hm

    meta_data = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep="\t")
    data_mat = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep="\t")

    start = time()
    ho = hm.run_harmony(data_mat, meta_data, ['donor'])
    end = time()
    logging.info("{:.2f} seconds elapsed".format(end - start))


@pytest.mark.parametrize("test_input", [1, 0])
def test_in_separate_processes(test_input):
    run_test_in_process(run_it, test_input)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
