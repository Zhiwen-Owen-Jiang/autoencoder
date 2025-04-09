import os
import gzip
import bz2
import logging
import numpy as np
import pandas as pd
from functools import reduce
from scipy.linalg import cho_solve, cho_factor


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


def inv(A):
    """
    Computing inverse for a symmetric and positive-definite matrix

    """
    cho_factors = cho_factor(A)
    A_inv = cho_solve(cho_factors, np.eye(A.shape[0]))

    return A_inv