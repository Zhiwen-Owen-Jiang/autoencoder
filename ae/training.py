import os
import time
import argparse
import traceback

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
    
from dsets import ImageDataset
from model import Autoencoder
# from heig.fpca import LocalLinear
from utils import *


"""
TODO: introduce kernel smoothing
1. smooth the target voxels, input raw images, recover smooth voxels (Denoise AE)
2. incorporate smoothing matrix in loss function, 
    2.1 compute nearest neighbors
    2.2 incorporate distance as weight in loss function
    2.3 bandwidth is a hyperparameter to fine tune

"""


class ReconstructionLoss:
    def __init__(self, model, lambda_l2=0):
        self.model = model
        self.lambda_l2 = lambda_l2
        self.mse_loss = nn.MSELoss(reduction="mean")

    def compute_masked_mse(self, recons, target, mask):
        recons = recons.view(-1)
        # masked_target = target[mask]
        masked_target = target.view(-1)[mask.view(-1)]
        return self.mse_loss(recons, masked_target)
    
    def compute_l2_penalty(self):
        l2_penalty = 0
        for param in self.model.parameters():
            l2_penalty += torch.sum(param ** 2)
        return l2_penalty
    
    def __call__(self, recons, target, mask):
        masked_mse_loss = self.compute_masked_mse(recons, target, mask)
        if self.lambda_l2 > 0:
            l2_penalty = self.lambda_l2 * self.compute_l2_penalty()
        else:
            l2_penalty = 0
        total_loss = masked_mse_loss + l2_penalty
        return total_loss


class ReconstructionCorr:
    def __init__(self, model):
        self.model = model
        
    def compute_masked_corr(self, recons, target, mask):
        recons = recons.view(-1)
        # masked_target = target[mask]
        masked_target = target.view(-1)[mask.view(-1)]
        stacked = torch.stack([recons, masked_target], dim=0)
        corr = torch.corrcoef(stacked)[0, 1]
        return corr
    
    def __call__(self, recons, target, mask):
        masked_corr = self.compute_masked_corr(recons, target, mask)
        return masked_corr


class VoxelCorrLoss:
    def __init__(self, model, lambda_corr=0.1):
        self.model = model
        self.lambda_corr = lambda_corr
    
    def compute_masked_corr(self, recons, target, mask):
        mask_3d = mask[0, 0]
        target = target.squeeze(1)[:, mask_3d]
        target_centered = target - target.mean(dim=0, keepdim=True)
        recons_centered = recons - recons.mean(dim=0, keepdim=True)
        cov = (target_centered * recons_centered).mean(dim=0)
        std_target = target_centered.std(dim=0) + 1e-8  # numerical stability
        std_recon = recons_centered.std(dim=0) + 1e-8
        corr_per_voxel = cov / (std_target * std_recon)
        mean_corr = corr_per_voxel.mean()

        return (self.lambda_corr * (1 - mean_corr).clamp(min=0, max=2))
    
    def __call__(self, recons, target, mask):
        return self.compute_masked_corr(recons, target, mask)


class Training(L.LightningModule):
    def __init__(self, input_shape, latent_dim, n_voxels, lambda_l2=0, lambda_corr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = Autoencoder(input_shape, latent_dim, n_voxels)
        self.loss_func = ReconstructionLoss(self.model, lambda_l2)
        self.corr_func = ReconstructionCorr(self.model)
        # self.voxel_corr_loss_func = VoxelCorrLoss(self.model, lambda_corr)

    def forward(self, inputs):
        image, mask = inputs
        recons, latent = self.model(image)
        return recons, latent, mask

    def training_step(self, batch, batch_idx):
        image, mask = batch
        recons, _ = self.model(image)
        recons_loss = self.loss_func(recons, image, mask)
        # corr_loss = self.voxel_corr_loss_func(recons, image, mask)
        loss = recons_loss
        corr = self.corr_func(recons, image, mask)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log("train_recons_loss", recons_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_corr", corr, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        image, mask = batch
        recons, _ = self.model(image)
        recons_loss = self.loss_func(recons, image, mask)
        # corr_loss = self.voxel_corr_loss_func(recons, image, mask)
        loss = recons_loss
        corr = self.corr_func(recons, image, mask)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        # self.log("val_recons_loss", recons_loss, prog_bar=True, sync_dist=True)
        self.log("val_corr", corr, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                "min",
                patience=4,
                min_lr=0.001 / 1000,
                factor=0.5,
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
    
    def on_after_backward(self):
        # Log gradient norms after the backward pass
        total_norm = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(2) # L2 norm of gradients
                total_norm += param_norm ** 2
                self.log(f"grad_norm_{name}", param_norm, prog_bar=False, logger=True)

        total_norm = total_norm ** 0.5 # Square root of summed norms
        self.log("grad_total_norm", total_norm, prog_bar=True, logger=True)


def data_split(images, train_prop=0.8, seed=42):
    # torch.manual_seed(seed)
    train_size = int(images.n_sub * train_prop)
    val_size = images.n_sub - train_size
    train_data, val_data = random_split(images, [train_size, val_size])

    return train_data, val_data
    

def check_input(args):
    if args.image is None:
        raise ValueError("--image is required")
    if args.out is None:
        raise ValueError("--out is required")
    
    check_existence(args.image)

    if args.keep is not None:
        args.keep = split_files(args.keep)
        args.keep = read_keep(args.keep)
        log.info(f"{len(args.keep)} subject(s) in --keep (logical 'and' for multiple files).")

    if args.remove is not None:
        args.remove = split_files(args.remove)
        args.remove = read_remove(args.remove)
        log.info(f"{len(args.remove)} subject(s) in --remove (logical 'or' for multiple files).")


def main(args, log):
    check_input(args)

    try:
        images = ImageDataset(args.image, norm=True)
        n_voxels = images.mask.sum().item()
        log.info(f"image shape: {images.shape}, {n_voxels} target voxels.")

        images.keep_and_remove(args.keep, args.remove)
        train_data, val_data = data_split(images)

        train_dl = DataLoader(
            train_data, 
            batch_size=32, 
            num_workers=2,
            pin_memory=True, 
            shuffle=True
        )
        val_dl = DataLoader(
            val_data, 
            batch_size=1, 
            num_workers=2,
            pin_memory=True, 
            shuffle=False
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        model_checkpoint = ModelCheckpoint(
            dirpath=args.out + "/model",
            monitor="val_loss",
            save_last=True,
            filename="{epoch}-{lr:.6f}-{train_loss:.6f}-{val_loss:.6f}-{train_corr:.6f}-{val_corr:.6f}",
            save_top_k=3,
        )

        tb_logger = TensorBoardLogger(save_dir=args.out + "/tb_logs")
        csv_logger = CSVLogger(save_dir=args.out + "/csv_logs")
        # pb = ProgressBar()

        # main training
        training_app = Training(images.shape, args.latent_dim, n_voxels, args.lambda_l2)
        trainer = L.Trainer(
            logger=[tb_logger, csv_logger],
            callbacks=[lr_monitor, model_checkpoint],
            sync_batchnorm=True,
            log_every_n_steps=10,
            accelerator="gpu",
            devices=2,
            benchmark=True,
            max_epochs=200
        )
        
        if args.check_point is not None:
            log.info(f"Read check point from {args.check_point}")
            trainer.fit(
                training_app, train_dataloaders=train_dl, val_dataloaders=val_dl, 
                ckpt_path=args.check_point
            )
        else:
            trainer.fit(
                training_app, train_dataloaders=train_dl, val_dataloaders=val_dl
            )
    finally:
        images.close()


parser = argparse.ArgumentParser()
parser.add_argument("--image", help="directory to imputed images in HDF5 format.")
parser.add_argument("--latent-dim", type=int, help="latent dimension")
parser.add_argument("--lambda-l2", type=float, help="lambda of L2 penalty")
parser.add_argument("--check-point")
parser.add_argument("--keep")
parser.add_argument("--remove")
parser.add_argument("--threads", type=int, help="number of threads.")
parser.add_argument("--out", help="output (prefix).")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out is None:
        args.out = "training"

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(""))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "training.py \\\n"
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