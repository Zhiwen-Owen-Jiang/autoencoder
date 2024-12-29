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
from tqdm import tqdm
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

    def __init__(self, image_file):
        """
        Parameters:
        ------------
        image_file: a image HDF5 file path. Images have been imputed and converted into 3D
        
        """
        self.file = h5py.File(image_file, "r")
        self.images = self.file["images"]
        ids = self.file["id"][:]
        self.ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        self.n_sub, *self.shape = self.images.shape
        self.id_idxs = np.arange(len(self.ids))
        self.extracted_ids = self.ids
        self.mask = torch.tensor(self.images[0] > 0, dtype=torch.float32)

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

        return image_t, self.mask
    
    def close(self):
        self.file.close()


class ImageImputation:
    def __init__(self, image_file, mask, out, crop=False, threads=1):
        self.file = h5py.File(image_file, "r")
        self.images = self.file["images"]
        self.coord = self.file["coord"][:]
        ids = self.file["id"][:]
        self.ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        self.n_sub = self.images.shape[0]

        self.mask = mask
        if crop:
            self.image_shape, (self.z_range, self.y_range, self.x_range) = self._crop_image(mask)
        else:
            self.image_shape = mask.shape
            self.z_range = slice(0, mask.shape[0])
            self.y_range = slice(0, mask.shape[1])
            self.x_range = slice(0, mask.shape[2])
        
        self.idxs_to_impute, self.idxs_res = self._idxs_to_impute()
        self.out = out
        self.threads = threads

        with h5py.File(f"{self.out}_imputed_images.h5", "w") as h5f:
            new_images = h5f.create_dataset(
                "images", shape=(self.n_sub, *self.image_shape), dtype="float32"
            )
            new_images.attrs["id"] = "id"
            new_images.attrs["coord"] = "coord"
            h5f.create_dataset("id", data=np.array(ids.tolist(), dtype="S10"))
            h5f.create_dataset("coord", data=self.coord)

    def _idxs_to_impute(self):
        """
        coord must be int
        
        """
        idxs_temp = np.where(self.mask == 1)
        idxs_temp = set(zip(*idxs_temp))
        idxs_res = tuple(zip(*self.coord.T))
        idxs_to_impute = idxs_temp.difference(idxs_res)
        idxs_res = np.array(idxs_res)
        idxs_to_impute = np.array(list(idxs_to_impute))

        return idxs_to_impute, idxs_res
    
    def get_nearest_neighbors(self, nearest_point=None):
        if nearest_point is None:
            self.nearest_point = {
                tuple(idx): self._get_nearest_point(idx) for idx in self.idxs_to_impute
            }
            pickle.dump(nearest_point, open(f"{self.out}_nn.dat", "wb"))
        else:
            self.nearest_point = nearest_point

    def _get_nearest_point(self, target):
        dis = [np.sum((target - idx) ** 2) for idx in self.idxs_res]
        return np.argmin(dis)
    
    def impute(self):
        original_coord = tuple(zip(*self.coord))
        impute_coord = tuple(zip(*np.array(list(self.nearest_point.keys()))))
        for idx, image in tqdm(enumerate(self.images), desc=f"{self.n_sub} images"):
            self.mask[original_coord] = image # (N, )
            self.mask[impute_coord] = image[list(self.nearest_point.values())]
            with h5py.File(f"{self.out}_imputed_images.h5", "r+") as h5f:
                h5f["images"][idx] = self.mask[self.z_range, self.y_range, self.x_range]

    @staticmethod
    def _crop_image(image, margin=5):
        """
        Crops a 3D image around the non-zero region with an additional margin.
        
        Parameters:
        ------------
        image (numpy.ndarray): Input 3D image of shape (D, H, W).
        margin (int): Number of voxels to include as margin around the non-zero region.
        
        Returns:
        ---------
        tuple: The slice indices used for cropping.

        """
        # Find the indices of the non-zero elements
        non_zero_indices = np.argwhere(image != 0)
        
        # Get the bounding box of the non-zero region
        z_min, y_min, x_min = non_zero_indices.min(axis=0)
        z_max, y_max, x_max = non_zero_indices.max(axis=0)
        
        # Apply margins
        z_min = max(z_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_min = max(x_min - margin, 0)
        
        z_max = min(z_max + margin + 1, image.shape[0])
        y_max = min(y_max + margin + 1, image.shape[1])
        x_max = min(x_max + margin + 1, image.shape[2])
        
        # Crop the image
        # cropped_image = image[z_min:z_max, y_min:y_max, x_min:x_max]
        shape = (z_max - z_min, y_max - y_min, x_max - x_min)
        cropped_range = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))

        return shape, cropped_range

    def close(self):
        self.file.close()


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
    try:
        image_imputation = ImageImputation(args.image, mask, args.out, crop=args.crop, threads=args.threads)
        image_imputation.get_nearest_neighbors(nearest_point)
        image_imputation.impute()
    finally:
        image_imputation.close()
    log.info(f"Save the imputed images to {args.out}_imputed_images.h5")


parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Directory to processed raw images in HDF5 format.")
parser.add_argument("--mask", help="a mask file (e.g., .nii.gz) as template.")
parser.add_argument("--nn", help="a dat file for nearest neighbor information.")
parser.add_argument("--threads", type=int, help="number of threads.")
parser.add_argument("--crop", action="store_true", help="if cropping image to remove unnecessary background.")
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