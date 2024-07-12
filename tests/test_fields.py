from io import StringIO

import numpy as np
import pandas as pd
import pytest

from chromo.polymers import Chromatin
from chromo.fields import UniformDensityField, get_neighboring_bins
import chromo.binders


def test_uniform_density_field_roundtrip():
    binders = [chromo.binders.get_by_name('HP1')]
    binders = chromo.binders.make_binder_collection(binders)

    p = Chromatin.straight_line_in_x(
        'Chr-1', np.ones(10), states=np.zeros((10, 1), dtype=int),
        binder_names=np.array(['HP1']),
        chemical_mods=np.zeros((10, 1), dtype=int),
        chemical_mod_names=np.array(['H3K9me3'])
    )

    udf = UniformDensityField([p], binders, 10, 20, 30, 40, 50, 60)
    udf_round_trip = UniformDensityField.from_file(
        StringIO(udf.to_file(None)), [p], binders
    )

    assert udf == udf_round_trip

    # ensure the code errors if the wrong number of polymers or binders is passed
    with pytest.raises(ValueError):
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [], binders
        )
    with pytest.raises(ValueError):
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p, p], binders
        )
    with pytest.raises(ValueError):
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p], pd.concat([binders, binders])
        )

    # should also error if the poly/binder has the wrong name
    with pytest.raises(ValueError):
        p2 = Chromatin.straight_line_in_x(
            'Chr-1', np.ones(10), states=np.zeros((10,)),
            binder_names=np.array(['HP1'])
        )
        p2.name = "different"
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p2], binders
        )
    with pytest.raises(ValueError):
        binders2 = chromo.binders.make_binder_collection(binders)
        binders2.loc[0, 'name'] = 'different'
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p], binders2
        )


def test_get_neighbors_at_ind():
    """For a 5-by-5-by-5 grid of voxels, we know what neighbors to expect.
    """
    nx = 5
    ny = 5
    nz = 5
    nbr_bins = get_neighboring_bins(nx, ny, nz)

    expected_neighbors_bin_12 = [
        6, 7, 8, 11, 12, 13, 16, 17, 18,
        106, 107, 108, 111, 112, 113, 116, 117, 118,
        31, 32, 33, 36, 37, 38, 41, 42, 43
    ]
    for nbr_bin in expected_neighbors_bin_12:
        assert nbr_bin in nbr_bins[12]

    expected_neighbors_bin_0 = [
        24, 20, 21, 4, 0, 1, 9, 5, 6,
        29, 25, 26, 34, 30, 31, 49, 45, 46,
        104, 100, 101, 109, 105, 106, 120, 121, 124
    ]
    for nbr_bin in expected_neighbors_bin_0:
        assert nbr_bin in nbr_bins[0]

    expected_neighbors_bin_24 = [
        18, 19, 15, 23, 24, 20, 3, 4, 0,
        28, 29, 25, 43, 44, 40, 48, 49, 45,
        118, 119, 115, 123, 124, 120, 103, 104, 100
    ]
    for nbr_bin in expected_neighbors_bin_24:
        assert nbr_bin in nbr_bins[24]

    for i in nbr_bins.keys():
        for j in nbr_bins[i]:
            assert i in nbr_bins[j]
