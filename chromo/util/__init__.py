from pathlib import Path
import numpy as np


_util_folder = Path(__file__).parent.absolute()
dss_params = np.loadtxt(_util_folder / Path("dssWLCparams"))


def combine_repeat(a, idx):
    """
    Combine the repeat entries in an array.

    :param a:
    :return: a_combine, idx
    """""
    #    a = np.array([[11, 2], [11, 3], [13, 4], [10, 10], [10, 1]])
    #    b = a[np.argsort(a[:, 0])]
    #    grps, idx = np.unique(b[:, 0], return_index=True)
    #    counts = np.add.reduceat(b[:, 1:], idx)
    #    print(np.column_stack((grps, counts)))

    # b = idx[np.argsort(idx)]
    b = np.sort(idx, kind='mergesort')
    grps, idx = np.unique(b, return_index=True)
    counts = np.add.reduceat(a, idx)
    a_with_index = np.column_stack((grps, counts))
    a_combine = a_with_index[:, 1:]
    idx_combine = a_with_index[:, 0].astype(int)

    return a_combine, idx_combine
