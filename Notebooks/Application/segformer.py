#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig

import torch

import initialization as init
from hub_mixin import SMPHubMixin


#-----------------------------------------------------------------------#
#                          SegFormer Model                              #
#-----------------------------------------------------------------------#
class SegFormer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_labels=1, dropout_p=0.5):
        super(SegFormer, self).__init__()
        self.num_labels = num_labels

        # Initialize the SegFormer backbone
        config = SegformerConfig(
            num_labels=self.num_labels,
            hidden_sizes=[64, 128, 320, 512],
            depths=[3, 3, 9, 3],
            num_channels=in_channels,
            drop_path_rate=dropout_p
        )
        self.backbone = SegformerModel(config)
        
        # Final convolutional layer to map transformer features to desired output channels
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(config.hidden_sizes[-1], out_channels, kernel_size=1),
            nn.Sigmoid()  # Sigmoid activation for binary segmentation tasks
        )
        
    def forward(self, x):
        # Pass input through SegFormer backbone
        features = self.backbone(pixel_values=x).last_hidden_state
        
        # Reshape features from (B, H*W, C) to (B, C, H, W)
        B, _, C = features.shape
        H, W = int(x.shape[-2] / 4), int(x.shape[-1] / 4)  # Adjust based on actual downsampling
        features = features.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Pass through segmentation head to get final output
        out = self.segmentation_head(features)
        
        # Upsample the output to match the input image size
        out = nn.functional.interpolate(out, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        
        return out