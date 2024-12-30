import os
import time
import argparse
import traceback
import h5py
import concurrent.futures
from filelock import FileLock
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
from utils import *


class ImageReader:
    """
    A class of reading 3D NIFTI images

    1. Images can be saved in muliple directories
    2. Support keep and remove
    3. Check duplicated images and images with zero variance
    
    """
    def __init__(self, img_files, ids, coord_img_file, crop, qc, out_dir, threads=1):
        """
        Parameters:
        ------------
        img_files: a list of image files
        ids: a list of ids
        coord_img_file: coordinate image file path
        crop: if cropping images
        qc: if doing quality control
        out_dir: output path
        threads: number of threads
        
        """
        self.out_dir = out_dir
        self.logger = logging.getLogger(__name__)

        # read coordinate image
        image = nib.load(coord_img_file).get_fdata()
        if crop:
            self.image_shape, (self.z_range, self.y_range, self.x_range) = self._crop_image(image)
        else:
            self.image_shape = image.shape
            self.z_range = slice(0, image.shape[0])
            self.y_range = slice(0, image.shape[1])
            self.x_range = slice(0, image.shape[2])    
        
        self.coord = np.stack(np.nonzero(image)).T
        self.n_voxels = self.coord.shape[0]
        if qc:
            self.img_files, self.ids = self._quality_check(img_files, ids, threads)
        else:
            self.img_files, self.ids = img_files, ids
        self.ids = pd.MultiIndex.from_arrays(
            [self.ids, self.ids], names=["FID", "IID"]
        )
        self.n_images = len(self.img_files)

    def _quality_check(self, img_files, ids, threads):
        """
        Checking image quality
        1. Images with exceeding zeros
        2. Images with zero variance
        
        """
        self.logger.info("Checking image quality ...")
        good_img_files = list()
        good_ids = list()

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(self._quality_check_, idx, img_file): idx
                for idx, img_file in enumerate(img_files)
            }
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{len(futures)} images",
            ):
                pass

            for future in concurrent.futures.as_completed(futures):
                try:
                    idx = future.result()
                    if idx is not None:
                        good_img_files.append(img_files[idx])
                        good_ids.append(ids[idx])
                except Exception as exc:
                    raise RuntimeError(f"Computation terminated due to error: {exc}")
                
        return good_img_files, good_ids

    def _quality_check_(self, idx, img_file):
        image = self._read_image(img_file)
        if (image > 0).sum() < 0.95 * self.n_voxels:
            self.logger.info(f"{img_file} is an invalid image with exceeding 0s.")
            return None
        if np.std(image[self.z_range, self.y_range, self.x_range]) == 0:
            self.logger.info(f"{img_file} is an invalid image with variance 0.")
            return None
        return idx

    def create_dataset(self):
        """
        Creating a HDF5 file saving images, coordinates, and ids

        """
        with h5py.File(self.out_dir, "w") as h5f:
            images = h5f.create_dataset(
                "images", shape=(self.n_images, *self.image_shape), dtype="float32"
            )
            h5f.create_dataset("id", data=np.array(self.ids.tolist(), dtype="S10"))
            h5f.create_dataset("coord", data=self.coord)
            images.attrs["id"] = "id"
            images.attrs["coord"] = "coord"
        self.logger.info(
            (
                f"{self.n_images} subjects and {self.n_voxels} voxels (vertices) "
                f"of shape {self.image_shape} in the imaging data."
            )
        )
        
    def read_save_image(self, threads):
        """
        Reading and saving images in parallel

        """
        self.logger.info("Reading images ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(self._read_save_image, idx, img_file)
                for idx, img_file in enumerate(self.img_files)
            ]
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{len(futures)} images",
            ):
                pass

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    executor.shutdown(wait=False)
                    if os.path.exists(f"{self.out_dir}.lock"):
                        os.remove(f"{self.out_dir}.lock")
                    if os.path.exists(self.out_dir):
                        os.remove(self.out_dir)
                    raise RuntimeError(f"Computation terminated due to error: {exc}")

        self.logger.info("Done.")
        if os.path.exists(f"{self.out_dir}.lock"):
            os.remove(f"{self.out_dir}.lock")

    def _read_save_image(self, idx, img_file):
        """
        Reading and writing a single image

        """
        image = self._read_image(img_file)
        lock_file = f"{self.out_dir}.lock"
        with FileLock(lock_file):
            with h5py.File(self.out_dir, "r+") as h5f:
                h5f["images"][idx] = image

    def _read_image(self, img_file):
        try:
            img = nib.load(img_file)
            data = img.get_fdata()
            return data[self.z_range, self.y_range, self.x_range]
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"cannot read {img_file}, did you provide a wrong NIFTI image?")

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
    

def get_image_list(img_dirs, suffixes, log, keep_idvs=None, remove_idvs=None):
    """
    Getting file path of images from multiple directories.

    Parameters:
    ------------
    img_dirs: a list of directories
    suffixes: a list of suffixes of images
    log: a logger
    keep_idvs: a pd.MultiIndex instance of IDs (FID, IID)
    remove_idvs: a pd.MultiIndex instance of IDs (FID, IID)

    Returns:
    ---------
    ids: a pd.MultiIndex instance of IDs
    img_files_list: a list of image files to read

    """
    img_files = {}
    n_dup = 0

    for img_dir, suffix in zip(img_dirs, suffixes):
        for img_file in os.listdir(img_dir):
            img_id = img_file.replace(suffix, "")
            if img_file.endswith(suffix) and (
                (keep_idvs is not None and img_id in keep_idvs) or (keep_idvs is None)
            ):
                if (remove_idvs is not None and img_id not in remove_idvs) or (remove_idvs is None):
                    if img_id in img_files:
                        n_dup += 1
                    else:
                        img_files[img_id] = os.path.join(img_dir, img_file)
    img_files = dict(sorted(img_files.items()))
    # ids = pd.MultiIndex.from_arrays(
    #     [img_files.keys(), img_files.keys()], names=["FID", "IID"]
    # )
    ids = list(img_files.keys())
    img_files_list = list(img_files.values())
    if n_dup > 0:
        log.info(f"WARNING: {n_dup} duplicated subject(s). Keep the first one.")

    return ids, img_files_list


def check_input(args):
    if args.image_dir is None:
        raise ValueError("--image-dir is required")
    if args.image_suffix is None:
        raise ValueError("--image-suffix is required")
    if args.coord_dir is None:
        raise ValueError("--coord-dir is required")
    if args.out is None:
        raise ValueError("--out is required")
    
    args.image_dir = args.image_dir.split(",")
    args.image_suffix = args.image_suffix.split(",")
    if len(args.image_dir) != len(args.image_suffix):
        raise ValueError("--image-dir and --image-suffix do not match")
    for image_dir in args.image_dir:
        check_existence(image_dir)
    check_existence(args.coord_dir)


def main(args, log):
    check_input(args)
    
    ids, img_files = get_image_list(
        args.image_dir, args.image_suffix, log, args.keep, args.remove
    )

    if len(img_files) == 0:
        raise ValueError(
            f"no image in {args.image_dir} with suffix {args.image_suffix}"
        )
    
    out_dir = f"{args.out}_images.h5"
    img_reader = ImageReader(img_files, ids, args.coord_dir, args.crop, args.qc, out_dir, args.threads)
    img_reader.create_dataset()
    img_reader.read_save_image(args.threads)

    log.info(f"\nSave the images to {out_dir}")


parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", help="Directory to processed raw images in HDF5 format.")
parser.add_argument("--image-suffix", help="a mask file (e.g., .nii.gz) as template.")
parser.add_argument("--coord-dir", help="a dat file for nearest neighbor information.")
parser.add_argument("--keep")
parser.add_argument("--remove")
parser.add_argument("--qc", action="store_true", help="if checking image quality")
parser.add_argument("--threads", type=int, help="number of threads.")
parser.add_argument("--crop", action="store_true", help="if cropping image to remove unnecessary background.")
parser.add_argument("--out", help="output (prefix).")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out is None:
        args.out = "image"

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(""))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "/work/users/o/w/owenjf/image_genetics/methods/autoencoder/ae/image.py \\\n"
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