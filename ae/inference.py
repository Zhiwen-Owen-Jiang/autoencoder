import os
import time
import argparse
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training import Training
from dsets import ImageDataset
from utils import *
import dataset as ds


def projection_ldr(ldr, covar):
    """
    Computing S'(I - M)S/n = S'S - S'X(X'X)^{-1}X'S/n,
    where I is the identity matrix,
    M = X(X'X)^{-1}X' is the project matrix for X,
    S is the LDR matrix.

    Parameters:
    ------------
    ldr (n, r): low-dimension representaion of imaging data
    covar (n, p): covariates, including the intercept

    Returns:
    ---------
    ldr_cov: variance-covariance matrix of LDRs

    """
    n = ldr.shape[0]
    inner_ldr = np.dot(ldr.T, ldr)
    inner_covar = np.dot(covar.T, covar)
    inner_covar_inv = inv(inner_covar)
    ldr_covar = np.dot(ldr.T, covar)
    part2 = np.dot(np.dot(ldr_covar, inner_covar_inv), ldr_covar.T)
    ldr_cov = (inner_ldr - part2) / n
    ldr_cov = ldr_cov.astype(np.float32)

    return ldr_cov


def check_input(args):
    if args.image is None:
        raise ValueError("--image is required")
    if args.covar is None:
        raise ValueError("--covar is required")
    if args.out is None:
        raise ValueError("--out is required")
    if args.check_point is None:
        raise ValueError("--check-point is required")
    
    if args.keep is not None:
        args.keep = split_files(args.keep)
        args.keep = read_keep(args.keep)
        log.info(f"{len(args.keep)} subject(s) in --keep (logical 'and' for multiple files).")

    if args.remove is not None:
        args.remove = split_files(args.remove)
        args.remove = read_remove(args.remove)
        log.info(f"{len(args.remove)} subject(s) in --remove (logical 'or' for multiple files).")
    

def compute_masked_corr(recons, target, mask):
    recons = recons.view(-1)
    masked_target = target[mask]
    stacked = torch.stack([recons, masked_target], dim=0)
    corr = torch.corrcoef(stacked)[0, 1]
    return corr


def main(args, log):
    check_input(args)

    try:
        # read images
        images = ImageDataset(args.image, norm=True)
        n_voxels = images.mask.sum().item()
        log.info(f"image shape: {images.shape}, {n_voxels} target voxels.")

        # read covariates
        log.info(f"Read covariates from {args.covar}")
        covar = ds.Covar(args.covar, args.cat_covar_list)

        # keep common subjects
        common_idxs = ds.get_common_idxs(images.ids, covar.data.index, args.keep)
        common_idxs = ds.remove_idxs(common_idxs, args.remove)
        images.keep_and_remove(common_idxs)
        log.info(f"{len(common_idxs)} common subjects in these files.")

        covar.keep_and_remove(common_idxs)
        covar.cat_covar_intercept()
        log.info(
            f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept)."
        )

        # create datasets
        inference_dl = DataLoader(
            images, 
            batch_size=1, 
            num_workers=2,
            pin_memory=True, 
            shuffle=False
        )
        
        # load model
        log.info(f"Read check point from {args.check_point}")
        model = Training.load_from_checkpoint(args.check_point)
        model.eval()
        model = model.model
        decoder_weights = model.output_layer.weight.data.cpu().numpy() # (N, r)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # inference
        log.info("Computing latent features ...")
        latents = list()
        corr = list()
        for image, mask, image_std_t in tqdm(inference_dl):
            image = image.to(device)
            with torch.no_grad():
                output, latent = model(image)
            latents.append(np.squeeze(latent.cpu().numpy()))
            corr.append(compute_masked_corr(output, image, mask).cpu().numpy())
        ldrs = np.array(latents)
        image_std = images.get_std()
        ldrs = image_std.reshape(-1, 1) * ldrs
        n_ldrs = ldrs.shape[1]
        log.info(f"Reconstruction correlation: {np.mean(corr):.3f}({np.std(corr):.3f})")

        # var-cov matrix of projected LDRs
        ldr_cov = projection_ldr(ldrs, np.array(covar.data))
        log.info(
            f"Removed covariate effects from LDRs and computed variance-covariance matrix.\n"
        )

        # save the output
        ldr_df = pd.DataFrame(ldrs, index=images.extracted_ids)
        ldr_df.to_csv(f"{args.out}_ldr_top{n_ldrs}.txt", sep="\t")
        np.save(f"{args.out}_ldr_cov_top{n_ldrs}.npy", ldr_cov)
        np.save(f"{args.out}_bases.npy", decoder_weights)

        corr_df = pd.DataFrame(corr, index=images.extracted_ids)
        corr_df.to_csv(f"{args.out}_recon_corr_top{n_ldrs}.txt", sep="\t")

        log.info(f"Saved the raw LDRs to {args.out}_ldr_top{n_ldrs}.txt")
        log.info(
            (
                f"Saved the variance-covariance matrix of covariate-effect-removed LDRs "
                f"to {args.out}_ldr_cov_top{n_ldrs}.npy"
            )
        )
        log.info(f"Saved the bases to {args.out}_bases.npy")
        log.info(f"Saved reconstruction correlations to {args.out}_recon_corr_top{n_ldrs}.txt")
        
    finally:
        images.close()


parser = argparse.ArgumentParser()
parser.add_argument("--image", help="directory to imputed images in HDF5 format.")
parser.add_argument("--covar", help="directory to covariate file")
parser.add_argument("--cat-covar-list")
parser.add_argument("--check-point")
parser.add_argument("--keep")
parser.add_argument("--remove")
parser.add_argument("--out", help="output (prefix).")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out is None:
        args.out = "inference"

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(""))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "inference.py \\\n"
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