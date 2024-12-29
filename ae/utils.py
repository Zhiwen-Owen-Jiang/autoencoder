import os
import gzip
import bz2
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
    

def split_files(arg):
    files = arg.split(",")
    for file in files:
        check_existence(file)
    return files


def read_keep(keep_files):
    """
    Extracting common subject IDs from multiple files
    All files are confirmed to exist
    Empty files are skipped without error/warning
    files w/ or w/o a header are ok
    Error out if no common IDs exist

    Parameters:
    ------------
    keep_files: a list of tab/white-delimited files

    Returns:
    ---------
    keep_idvs_: pd.MultiIndex of common subjects

    """
    for i, keep_file in enumerate(keep_files):
        if os.path.getsize(keep_file) == 0:
            continue
        _, compression = check_compression(keep_file)

        try:
            keep_idvs = pd.read_csv(
                keep_file,
                sep="\s+",
                header=None,
                usecols=[0, 1],
                dtype={0: str, 1: str},
                compression=compression,
            )
        except ValueError:
            raise ValueError('two columns FID and IID are required')
        
        keep_idvs = pd.MultiIndex.from_arrays(
            [keep_idvs[0], keep_idvs[1]], names=["FID", "IID"]
        )
        if i == 0:
            keep_idvs_ = keep_idvs.copy()
        else:
            keep_idvs_ = keep_idvs_.intersection(keep_idvs)

    if len(keep_idvs_) == 0:
        raise ValueError("no subjects are common in --keep")

    return keep_idvs_


def read_remove(remove_files):
    """
    Removing subject IDs from multiple files
    All files are confirmed to exist
    Empty files are skipped without error/warning
    files w/ or w/o a header are ok

    Parameters:
    ------------
    remove_files: a list of tab/white-delimited files

    Returns:
    ---------
    remove_idvs_: pd.MultiIndex of common subjects

    """
    for i, remove_file in enumerate(remove_files):
        if os.path.getsize(remove_file) == 0:
            continue
        _, compression = check_compression(remove_file)

        try:
            remove_idvs = pd.read_csv(
                remove_file,
                sep="\s+",
                header=None,
                usecols=[0, 1],
                dtype={0: str, 1: str},
                compression=compression,
            )
        except ValueError:
            raise ValueError('two columns FID and IID are required')
        
        remove_idvs = pd.MultiIndex.from_arrays(
            [remove_idvs[0], remove_idvs[1]], names=["FID", "IID"]
        )
        if i == 0:
            remove_idvs_ = remove_idvs.copy()
        else:
            remove_idvs_ = remove_idvs_.union(remove_idvs, sort=False)

    return remove_idvs_


def check_compression(dir):
    """
    Checking which compression should use

    Parameters:
    ------------
    dir: diretory to the dataset

    Returns:
    ---------
    openfunc: function to open the file
    compression: type of compression

    """
    if dir.endswith("gz") or dir.endswith("bgz"):
        compression = "gzip"
        openfunc = gzip.open
    elif dir.endswith("bz2"):
        compression = "bz2"
        openfunc = bz2.BZ2File
    elif (
        dir.endswith("zip")
        or dir.endswith("tar")
        or dir.endswith("tar.gz")
        or dir.endswith("tar.bz2")
    ):
        raise ValueError(
            "files with suffix .zip, .tar, .tar.gz, .tar.bz2 are not supported"
        )
    else:
        openfunc = open
        compression = None

    return openfunc, compression