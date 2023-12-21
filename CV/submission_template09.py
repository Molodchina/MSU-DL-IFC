import torch
import torch.nn as nn
import torch.nn.functional as F

def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels,
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''

    # Реализуйте блок вида conv -> relu -> max_pooling.
    # Параметры слоя conv заданы параметрами функции encoder_block.
    # MaxPooling должен быть с ядром размера 2.
    block = nn.Sequential(
        # ВАШ КОД ТУТ
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    return block

def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels,
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''

    # Реализуйте блок вида conv -> relu -> upsample.
    # Параметры слоя conv заданы параметрами функции encoder_block.
    # Upsample должен быть со scale_factor=2. Тип upsampling (mode) можно выбрать любым.
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest')
    )

    return block

class create_model(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        параметры:
            - in_channels: количество каналов входного изображения
            - out_channels: количество каналов выхода нейросети
        '''
        super().__init__()

        self.enc1_block = encoder_block(in_channels, 32, 7, 3)
        self.enc2_block = encoder_block(32, 64, 3, 1)
        self.enc3_block = encoder_block(64, 128, 3, 1)

        # поймите, какие параметры должны быть у dec1_block, dec2_block и dec3_block
        # dec1_block должен быть симметричен блоку enc3_block
        # dec2_block должен быть симметричен блоку enc2_block
        # но обратите внимание на skip connection между выходом enc2_block и входом dec2_block
        # (см что подается на вход dec2_block в функции forward)
        # какое количество карт активации будет тогда принимать на вход dec2_block?
        # также обратите внимание на skip connection между выходом enc1_block и входом dec3_block
        # (см что подается на вход dec3_block в функции forward)
        # какое количество карт активации будет тогда принимать на вход dec3_block?
        self.dec1_block = decoder_block(128, 64, 3, 1)
        self.dec2_block = decoder_block(128, 32, 3, 1)
        self.dec3_block = decoder_block(64, out_channels, 3, 1)

    def forward(self, x):

        # downsampling part
        enc1 = self.enc1_block(x)
        enc2 = self.enc2_block(enc1)
        enc3 = self.enc3_block(enc2)

        dec1 = self.dec1_block(enc3)
        # из-за skip connection dec2 должен принимать на вход сконкатенированные карты активации
        # из блока dec1 и из блока enc2.
        # конкатенация делается с помощью torch.cat
        dec2 = self.dec2_block(torch.cat([dec1, enc2], 1))
        # из-за skip connection dec3 должен принимать на вход сконкатенированные карты активации
        # из блока dec2 и из блока enc1.
        # конкатенация делается с помощью torch.cat
        dec3 = self.dec3_block(torch.cat([dec2, enc1], 1))

        return dec3
