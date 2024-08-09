#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
import torch.nn as nn
from collections import OrderedDict


#-----------------------------------------------------------------------#
#        2 Dimensional Implementation of UNet acrchitecture             #
# Reference:                                                            #
# Ronneberger, O.; Fischer, P.; Brox, T. U-Net: Convolutional Networks  #
# for Biomedical Image Segmentation. In Proceedings of the International#
# Conference on Medical image computing and computer-assisted           #
# intervention, Munich, Germany, 5–9 October 2015; pp. 234–241.         #
#-----------------------------------------------------------------------#
# Adopted from:                                                         #
# The net:                                                              #
# github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py #                                                 
# https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb     #
# The weight initialization:                                            #
# https://discuss.pytorch.org/t/pytorch-how-to-initialize-weights       #
#   /81511/4                                                            #
# https://discuss.pytorch.org/t                                         #
#   /how-are-layer-weights-and-biases-initialized-by-default/13073/24   #
# The weight standardization:                                           #
# https://github.com/joe-siyuan-qiao/WeightStandardization              #
# Siyuan Qiao et al., Micro-Batch Training with Batch-Channel           #
#      Normalization and Weight Standardization,arXiv:1903.10520v2,2020 #
#-----------------------------------------------------------------------#
# in_channels:   number of input channels                               #
# out_channels:  number of output channels                              #
# init_features: number of filters in the first encoding layer, it      #
#                doubles at the successive encoding steps and halves at #
#                each decoding layer.                                   #
# dropout_p:     dropout probability                                    #
# mean, std:     mean and standard deviation to be used for weight      #
#                initialization using Gaussian distribution.            #
#                The standard deviation is the square root of (2/N),    #
#                where N is the number of incoming nodes of one neuron. #   
#-----------------------------------------------------------------------#
class UNet_2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, dropout_p=0.5):
        super(UNet_2D, self).__init__()
        self.init_features = init_features
        
        # Encoder
        self.enc1 = self.residual_conv_block(in_channels, init_features)
        self.enc2 = self.residual_conv_block(init_features, init_features * 2)
        self.enc3 = self.residual_conv_block(init_features * 2, init_features * 4)
        self.enc4 = self.residual_conv_block(init_features * 4, init_features * 8)
        
        # Bottleneck
        self.bottleneck = self.residual_conv_block(init_features * 8, init_features * 16)
        
        # Decoder
        self.upconv4 = self.upconv(init_features * 16, init_features * 8)
        self.upconv3 = self.upconv(init_features * 8, init_features * 4)
        self.upconv2 = self.upconv(init_features * 4, init_features * 2)
        self.upconv1 = self.upconv(init_features * 2, init_features)
        
        # Final Layer
        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def residual_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.residual_conv_block(self.init_features * 16, self.init_features * 8)(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.residual_conv_block(self.init_features * 8, self.init_features * 4)(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.residual_conv_block(self.init_features * 4, self.init_features * 2)(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.residual_conv_block(self.init_features * 2, self.init_features)(dec1)
        
        # Final Layer
        out = self.final_conv(dec1)
        
        return out