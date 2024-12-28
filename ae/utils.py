import os
import logging
import pandas as pd
from functools import reduce


def GetLogger(logpath):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(logpath, mode="w")
    # fh.setLevel(logging.INFO)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)

    return log


def sec_to_str(t):
    """Convert seconds to days:hours:minutes:seconds"""
    [d, h, m, s, n] = reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24]
    )
    f = ""
    if d > 0:
        f += "{D}d:".format(D=d)
    if h > 0:
        f += "{H}h:".format(H=h)
    if m > 0:
        f += "{M}m:".format(M=m)

    f += "{S}s".format(S=s)
    return f


def get_common_idxs(*idx_list, single_id=False):
    """
    Getting common indices among a list of double indices for subjects.
    Each element in the list must be a pd.MultiIndex instance.

    Parameters:
    ------------
    idx_list: a list of pd.MultiIndex
    single_id: if return single id as a list

    Returns:
    ---------
    common_idxs: common indices in pd.MultiIndex or list

    """
    common_idxs = None
    for idx in idx_list:
        if idx is not None:
            if not isinstance(idx, pd.MultiIndex):
                raise TypeError("index must be a pd.MultiIndex instance")
            if common_idxs is None:
                common_idxs = idx.copy()
            else:
                common_idxs = common_idxs.intersection(idx)
    if common_idxs is None:
        raise ValueError("no valid index provided")
    if len(common_idxs) == 0:
        raise ValueError("no common index exists")
    if single_id:
        common_idxs = common_idxs.get_level_values("IID").tolist()

    return common_idxs


def remove_idxs(idx1, idx2, single_id=False):
    """
    Removing idx2 (may be None) from idx1
    idx1 must be a pd.MultiIndex instance.

    Parameters:
    ------------
    idxs1: a pd.MultiIndex of indices
    idxs2: a pd.MultiIndex of indices
    single_id: if return single id as a list

    Returns:
    ---------
    idxs: indices in pd.MultiIndex or list

    """
    if not isinstance(idx1, pd.MultiIndex):
        raise TypeError("index must be a pd.MultiIndex instance")
    if idx2 is not None:
        idx = idx1.difference(idx2)
        if len(idx) == 0:
            raise ValueError("no subject remaining after --remove")
    else:
        idx = idx1
    if single_id:
        idx = idx.get_level_values("IID").tolist()
    
    return idx


def check_existence(arg, suffix=""):
    """
    Checking file existence

    """
    if arg is not None and not os.path.exists(f"{arg}{suffix}"):
        raise FileNotFoundError(f"{arg}{suffix} does not exist")

