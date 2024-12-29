import os
import time
import argparse
import traceback
import nibabel as nib

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as L
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    
from .dsets import ImageDataset
from .model import Autoencoder
# from heig.fpca import LocalLinear
from utils import *


class ReconstructionLoss:
    def __init__(self, model, lambda_l2=0):
        self.model = model
        self.lambda_l2 = lambda_l2
        self.mse_loss = nn.MSELoss(reduction="mean")

    def compute_masked_mse(self, target, recons, mask):
        masked_recons = recons * mask
        masked_target = target * mask
        return self.mse_loss(masked_recons, masked_target)
    
    def compute_l2_penalty(self):
        l2_penalty = 0
        for param in self.model.parameters():
            l2_penalty += torch.sum(param ** 2)
        return l2_penalty
    
    def __call__(self, recons, target, mask):
        masked_mse_loss = self.compute_masked_mse(recons, target, mask)
        l2_penalty = self.lambda_l2 * self.compute_l2_penalty()
        total_loss = masked_mse_loss + l2_penalty
        return total_loss


class Training(L.LightningModule):
    def __init__(self, model, mask, lambda_l2):
        super().__init__()
        self.save_hyperparameters(["lambda_l2"])
        self.model = model
        self.mask = mask
        self.loss_func = ReconstructionLoss(self.model, lambda_l2)

    def forward(self, inputs):
        _, latent = self.model(inputs)
        return latent

    def training_step(self, batch, batch_idx):
        recons, _ = self.model(batch)
        loss = self.loss_func(recons, batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        recons, _ = self.model(batch)
        loss = self.loss_func(recons, batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001, **self.hparams.optimizer_hparams)
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


def data_split(images, train_prop=0.8, seed=42):
    torch.manual_seed(seed)
    train_size = int(images.n_sub * train_prop)
    val_size = images.n_sub - train_size
    train_data, val_data = random_split(images, [train_size, val_size])

    return train_data, val_data
    

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
    check_input(args, log)
    mask = nib.load(args.mask).get_fdata()
    images = ImageDataset(args.image, mask)
    train_data, val_data = data_split(images)

    train_dl = DataLoader(train_data, batch_size=32, pin_memory=True, shuffle=True)
    val_dl = DataLoader(val_data,  batch_size=32, pin_memory=True, shuffle=False)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    model_checkpoint = ModelCheckpoint(
        dirpath=args.out + "/model",
        monitor="val_loss",
        save_last=True,
        filename="{epoch}-{train_loss:.6f}-{val_loss:.6f}",
        save_top_k=5,
    )

    tb_logger = TensorBoardLogger(save_dir=args.out + "/tb_logs")
    csv_logger = CSVLogger(save_dir=args.out + "/csv_logs")
    pb = ProgressBar(refresh_rate=2)

    # main training
    model = Autoencoder(images.shape, args.latent_dim)
    training_app = Training(model, mask, args.lambda_l2)
    trainer = L.Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, model_checkpoint, pb],
        sync_batchnorm=True,
        log_every_n_steps=20,
        accelerator="auto",
        devices=4,
        benchmark=True,
        max_epochs=100,
    )

    trainer.fit(
        training_app, train_dataloaders=train_dl, val_dataloaders=val_dl
    )


parser = argparse.ArgumentParser()
parser.add_argument("--image", help="directory to imputed images in HDF5 format.")
parser.add_argument("--mask", help="a mask file (e.g., .nii.gz) as template.")
parser.add_argument("--latent-dim", type=int, help="latent dimension")
parser.add_argument("--lambda-l2", type=float, help="lambda of L2 penalty")
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