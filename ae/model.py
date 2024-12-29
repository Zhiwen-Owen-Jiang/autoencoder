import numpy as np
import torch
import torch.nn as nn


"""
An autoencoder to efficient representation learning
Encoder: 3D CNN of multiple layers and channels
Decoder: a single fully connected layer

1. Volumetric images: 3D CNN; surface-based images: GCN
2. images of fixed shape 

"""


class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.batchnorm = nn.BatchNorm3d(1)

        self.encoder = nn.Sequential(
            self.encoder_block(1, 16),
            nn.MaxPool3d(kernel_size=2, padding=1),
            self.encoder_block(16, 32),
            nn.MaxPool3d(kernel_size=2, padding=1),
            self.encoder_block(32, 64),
            nn.MaxPool3d(kernel_size=2, padding=1),
            self.encoder_block(64, 128),
            nn.MaxPool3d(kernel_size=2, padding=1),
            self.encoder_block(128, 256)
        )

        with torch.no_grad():
            x_dummy = torch.zeros(1, 1, *input_shape) # Dummy input to get dimensions
            encoded_dummy = self.encoder(x_dummy)
            self.flattened_size = encoded_dummy.numel()
        
        self.fc_enc = nn.Linear(self.flattened_size, latent_dim)
        flattened_image_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.output_layer = nn.Linear(latent_dim, flattened_image_size)

        self._init_weights()

    def encoder_block(self, input_channels, output_channels):
        encoder = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True)
        )

        return encoder

    def forward(self, input_batch):
        bn_output = self.batchnorm(input_batch)
        encoded = self.encoder(bn_output)
        encoded_flat = encoded.view(encoded.size(0), -1)
        latent = self.fc_enc(encoded_flat)
        output = self.output_layer(latent)
        output = output.view(-1, 1, *self.input_shape)

        return output, latent

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / np.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)