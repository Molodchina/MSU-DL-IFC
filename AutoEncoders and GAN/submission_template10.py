# import numpy as np
# import torch
from torch import nn
# from torch.nn import functional as F


def encoder_block(in_channels, out_channels, kernel_size, padding):
    """
    Блок, который принимает на вход карты активации с количеством каналов in_channels,
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    """
    # block = nn.Sequential(
    #     # ВАШ КОД ТУТ
    # )
    block = nn.Sequential(
        # ВАШ КОД ТУТ
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    return block


def decoder_block(in_channels, out_channels, kernel_size, padding):
    """
    Блок, который принимает на вход карты активации с количеством каналов in_channels,
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    """
    # block = nn.Sequential(
    #     # ВАШ КОД ТУТ
    # )
    block = nn.Sequential(
        # ВАШ КОД ТУТ
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear')
    )

    return block


#
class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=1):
        super().__init__()

        # encoder blocks
        self.encoder = nn.Sequential(
            encoder_block(in_channels, 32, kernel_size, padding),
            encoder_block(32, 64, kernel_size, padding)
        )

        # decoder blocks
        self.decoder = nn.Sequential(
            decoder_block(64, 32, kernel_size, padding),
            decoder_block(32, out_channels, kernel_size, padding)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


def create_model():
    return Autoencoder()
