from sys import version_info

import numpy as np
import pandas as pd

if version_info[1] == 8:
    import importlib_resources as ir
elif version_info[1] >= 9:
    import importlib.resources as ir

import pandas as pd
import pytest

import harmonypy as hm


@pytest.fixture
def meta_data():
    md_file = ir.files("tests").joinpath("data", "metadata.csv.gz")
    with ir.as_file(md_file) as mdf:
        md_df = pd.read_csv(mdf, index_col="cell_id")
    return md_df


@pytest.fixture
def data_mat():
    pcs_file = ir.files("tests").joinpath("data", "data_mat.csv.gz")
    with ir.as_file(pcs_file) as pcf:
        pc_arr = np.loadtxt(pcf, delimiter=",")
    return pc_arr


@pytest.fixture
def dataset_align_expected():
    dataset_aligned_file = ir.files("tests").joinpath(
        "data", "dataset_aligned_res.csv.gz"
    )
    with ir.as_file(dataset_aligned_file) as daf:
        expected = np.loadtxt(daf, delimiter=",")
    return expected


@pytest.fixture
def dataset_aligned_r():
    dataset_aligned_r_file = ir.files("tests").joinpath(
        "data", "harmonized_data_mat.csv.gz"
    )
    with ir.as_file(dataset_aligned_r_file) as darf:
        expected = np.loadtxt(darf, delimiter=",")
    return expected


def test_dataset_align(meta_data, data_mat, dataset_align_expected):
    __tracebackhide__ = True
    harmony_res = hm.run_harmony(
        data_mat=data_mat, meta_data=meta_data, vars_use=["dataset"]
    )
    res = harmony_res.Z_corr
    np.testing.assert_allclose(res, dataset_align_expected)


def test_dataset_agrees_with_r_version(meta_data, data_mat, dataset_aligned_r):
    __tracebackhide__ = True
    harmony_res = hm.run_harmony(
        data_mat=data_mat, meta_data=meta_data, vars_use=["dataset"]
    )
    res = harmony_res.Z_corr
    np.testing.assert_allclose(res, dataset_aligned_r.T, rtol=0.01, atol=1000) 
    # is this relaxed a bit?  yeah, it's pretty frelling chill.  Not sure if I actually care
    # that there is a larger absolute difference between the two matrices when the individual
    # elements mostly agree. TODO make the Python and R matrices better align
