import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn.functional as F


"""
An autoencoder to efficient representation learning
Encoder: 3D CNN of multiple layers and channels
Decoder: a single fully connected layer

1. Volumetric images: 3D CNN; surface-based images: GCN
2. images of fixed shape 

"""


class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim, n_voxels):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.n_voxels = n_voxels

        self.batchnorm = nn.BatchNorm3d(1)

        self.encoder = nn.Sequential(
            # self.encoder_block(1, 16),
            ResBlock3D(1, 16),
            nn.MaxPool3d(kernel_size=2, padding=1),
            # self.encoder_block(16, 32),
            ResBlock3D(16, 32),
            nn.MaxPool3d(kernel_size=2, padding=1),
            # self.encoder_block(32, 64),
            ResBlock3D(32, 64),
            nn.MaxPool3d(kernel_size=2, padding=1),
            # self.encoder_block(64, 128),
            ResBlock3D(64, 128),
            nn.MaxPool3d(kernel_size=2, padding=1),
            # self.encoder_block(128, 256)
            ResBlock3D(128, 256),
            nn.MaxPool3d(kernel_size=2, padding=1),
            ResBlock3D(256, 512),
        )

        with torch.no_grad():
            x_dummy = torch.zeros(1, 1, *input_shape) # Dummy input to get dimensions
            encoded_dummy = self.encoder(x_dummy)
            self.flattened_size = encoded_dummy.numel()
        
        self.fc_enc = nn.Linear(self.flattened_size, latent_dim)
        self.output_layer = nn.Linear(latent_dim, self.n_voxels, bias=False)

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

        return output, latent

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / np.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
        # If in_channels ≠ out_channels, use a projection to match shapes
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.projection is not None:
            identity = self.projection(x)
        out = self.conv_block(x)
        return F.leaky_relu(out + identity)
    

class Bottleneck3D(nn.Module):
    expansion = 4  # final out_channels = mid_channels * expansion

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet3DEncoder(nn.Module):
    def __init__(self, input_channels=1, block=Bottleneck3D, layers=[3, 4, 6, 3], latent_dim=256):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, latent_dim)

    def _make_layer(self, block, mid_channels, blocks, stride):
        downsample = None
        out_channels = mid_channels * block.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

        layers = [block(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, mid_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)  # [B, 256, D/4, H/4, W/4]
        x = self.layer2(x)  # [B, 512, ...]
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # [B, 2048, 1, 1, 1]
        x = torch.flatten(x, 1)  # [B, 2048]
        x = self.fc(x)           # [B, latent_dim]
        return x
    

class ResNet3DAutoEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim, n_voxels):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.n_voxels = n_voxels

        self.encoder = ResNet3DEncoder(latent_dim=self.latent_dim)
        self.output_layer = nn.Linear(self.latent_dim, self.n_voxels, bias=False)

        self._init_weights()
    
    def forward(self, input_batch):
        latent = self.encoder(input_batch)
        output = self.output_layer(latent)

        return output, latent

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / np.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


class GNNAutoencoder(nn.Module):
    def __init__(self, latent_dim, n_voxels):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_voxels = n_voxels

        self.batchnorm = BatchNorm(1)

        self.encoder = nn.Sequential(
            self.encoder_block(1, 16),
            self.encoder_block(16, 32),
            self.encoder_block(32, 64),
            self.encoder_block(64, 128),
            self.encoder_block(128, 256)
        )

        self.flattened_size = 256
        self.pool = global_mean_pool
        self.batch = torch.zeros(self.n_voxels, dtype=torch.long)
        self.fc_enc = nn.Linear(self.flattened_size, self.latent_dim)
        self.output_layer = nn.Linear(self.latent_dim, self.n_voxels, bias=False)

        self._init_weights()

    def encoder_block(self, input_channels, output_channels):
        encoder = nn.Sequential(
            GCNConv(input_channels, output_channels, bias=True),
            BatchNorm(output_channels),
            nn.LeakyReLU(inplace=True),
            GCNConv(output_channels, output_channels, bias=True),
            BatchNorm(output_channels),
            nn.LeakyReLU(inplace=True)
        )

        return encoder

    def forward(self, input_batch, edge_index):
        bn_output = self.batchnorm(input_batch)
        encoded = self.encoder(bn_output, edge_index)
        encoded_pool = self.pool(encoded, self.batch)
        latent = self.fc_enc(encoded_pool)
        output = self.output_layer(latent)

        return output, latent

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, GCNConv}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / np.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)