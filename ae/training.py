import logging
import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
    
from .dsets import ImageDataset
from .model import Autoencoder
# from heig.fpca import LocalLinear


class TrainingApp:
    def __init__(
            self, 
            input_shape, 
            latent_dim,
            epochs,
            batch_size,
            lr,
            num_workers,
        ):
        # training parameters
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.batch_size *= torch.cuda.device_count()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.initModel()
        self.optimizer, _ = self.initOptimizer()
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.logger = logging.getLogger(__name__)

    def initModel(self):
        model = Autoencoder()
        if self.use_cuda:
            self.logger.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                "min",
                patience=4,
                min_lr=self.hparams["lr"] / 1000,
                factor=0.5,
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return optimizer, lr_scheduler_config

    def initDl(self, is_train):
        ds = ImageDataset(

        )
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.use_cuda,
        )

        return dl

    def initTensorboardWriters(self):
        pass