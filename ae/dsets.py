import os
import time
import argparse
import traceback
import h5py
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from utils import *


"""
Creating a dataset to load imputed 3D images
Imputing images and converting to 3D

"""

class ImageDataset(Dataset):
    """
    Creating a image dataset for NIFTI images
    
    """

    def __init__(self, image_file, mask):
        """
        Parameters:
        ------------
        image_file: a image HDF5 file path. Images have been imputed and converted into 3D
        mask: an np.array of image mask
        
        """
        self.file = h5py.File(image_file, "r")
        self.images = self.file["images"]
        ids = self.file["id"][:]
        self.ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        self.n_sub = self.images.shape[0]
        self.id_idxs = np.arange(len(self.ids))
        self.extracted_ids = self.ids
        self.mask = mask

    def keep_and_remove(self, keep_idvs=None, remove_idvs=None, check_empty=True):
        """
        Keeping and removing subjects

        Parameters:
        ------------
        keep_idvs: subject indices in pd.MultiIndex to keep 
        remove_idvs: subject indices in pd.MultiIndex to remove
        check_empty: if check the current image set is empty

        """
        if keep_idvs is not None:
            self.extracted_ids = get_common_idxs(self.extracted_ids, keep_idvs)
        if remove_idvs is not None:
            self.extracted_ids = remove_idxs(self.extracted_ids, remove_idvs)
        if check_empty and len(self.extracted_ids) == 0:
            raise ValueError("no subject remaining after --keep and/or --remove")
        
        self.n_sub = len(self.extracted_ids)
        self.id_idxs = np.arange(len(self.ids))[self.ids.isin(self.extracted_ids)]

    def __len__(self):
        return self.n_sub
    
    def __getitem__(self, index):
        image = self.images[self.id_idxs[index]]
        image_t = torch.from_numpy(image).to(torch.float32)
        image_t = image_t.unsqueeze(0)
        
        return image_t
    
    def close(self):
        self.file.close()


def impute_by_nearest_point(image_file, mask, output, nearest_point=None, threads=1):
    """
    Imputing images by the nearest point
    
    """
    with h5py.File(image_file, "r") as file:
        images = file["images"]
        coord = file["coord"][:]
        ids = file["id"][:]
        ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        n_sub = images.shape[0]
        
        # extracting indices in the mask that need to impute
        idxs_temp = np.where(mask == 1)
        idxs_temp = set(zip(*idxs_temp))
        idxs_res = tuple(zip(*coord.T))
        idxs_to_impute = idxs_temp.difference(idxs_res)
        idxs_res = np.array(idxs_res)
        idxs_to_impute = np.array(list(idxs_to_impute))

        # creating a new image h5 file
        image_shape = mask.shape
        with h5py.File(f"{output}_imputed_images.h5", "w") as h5f:
            new_images = h5f.create_dataset(
                "images", shape=(n_sub, *image_shape), dtype="float32"
            )
            new_images.attrs["id"] = "id"
            new_images.attrs["coord"] = "coord"
            h5f.create_dataset("id", data=np.array(ids.tolist(), dtype="S10"))
            h5f.create_dataset("coord", data=coord)

        # imputing images by the nearest neighbor and save
        if nearest_point is None:
            nearest_point = get_nearest_point(idxs_res, idxs_to_impute, threads)
            pickle.dump(nearest_point, open(f"{output}_nn.dat", "wb"))
        for idx, image in enumerate(images):
            for target, point in nearest_point.items():
                mask[target] = image[point]
                with h5py.File(f"{output}_imputed_images.h5", "r+") as h5f:
                    h5f["images"][idx] = mask


def get_nearest_point(idxs_res, idxs_to_impute, threads):
    nearest_point = {
        tuple(idx): _get_nearest_point(idx, idxs_res) for idx in idxs_to_impute
    }
    return nearest_point


def _get_nearest_point(target, coord):
    dis = [np.sum((target - idx) ** 2) for idx in coord]
    return np.argmin(dis)


def check_input(args):
    if args.image is None:
        raise ValueError("--image is required")
    if args.mask is None:
        raise ValueError("--mask is required")
    if args.out is None:
        raise ValueError("--out is required")
    
    check_existence(args.image)
    check_existence(args.mask)


def main(args, log):
    check_input(args)
    mask = nib.load(args.mask).get_fdata()
    log.info(f"Read mask file from {args.mask}")

    if args.nn is not None:
        nearest_point = pickle.load(open(args.nn, "rb"))
        log.info(f"Read nearest neighbors from {args.nn}")
    else:
        log.info(f"No nearest neighbors provided, will be inferred from mask.")
        nearest_point = None

    log.info(f"Imputing images by the nearest neighbor ...")
    impute_by_nearest_point(args.image, mask, args.out, nearest_point, args.threads)
    log.info(f"Save the imputed images to {args.out}_imputed_images.h5")


parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Directory to processed raw images in HDF5 format.")
parser.add_argument("--mask", help="a mask file (e.g., .nii.gz) as template.")
parser.add_argument("--nn", help="a dat file for nearest neighbor information.")
parser.add_argument("--threads", type=int, help="number of threads.")
parser.add_argument("--out", help="output (prefix).")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out is None:
        args.out = "dsets"

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(""))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "dsets.py \\\n"
        options = [
            "--" + x.replace("_", "-") + " " + str(opts[x]) + " \\"
            for x in non_defaults
        ]
        header += "\n".join(options).replace(" True", "").replace(" False", "")
        header = header + "\n"
        log.info(header)
        main(args, log)
    except Exception:
        log.info(traceback.format_exc())
        raise
    finally:
        log.info(f"\nAnalysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")