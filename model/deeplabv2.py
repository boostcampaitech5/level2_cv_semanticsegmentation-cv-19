import torch
import torch.nn as nn
import torch.nn.functional as F

import constants
from model.base_model import BaseModel


# conv + bn + relu 단위
def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)


# Bottleneck for ResBlock
class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, downsample=False):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(in_ch, out_ch//4, 1, 1, 0),
            conv_block(out_ch//4, out_ch//4, 3, stride, dilation, dilation),
            conv_block(out_ch//4, out_ch, 1, 1, 0, relu=False)
        )
        self.downsample = nn.Sequential(
            conv_block(in_ch, out_ch, 1, stride, 0, 1, False)
        ) if downsample else None
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out


# ResBlock for ResNet101
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dilation, num_layers):
        super().__init__()
        block = []
        for i in range(num_layers):
            block.append(Bottleneck(
                in_ch if i==0 else out_ch, 
                out_ch, 
                stride if i==0 else 1, 
                dilation,
                True if i==0 else False
            ))
            
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


# ResNet101 backbone(conv1 ~ conv5 layer)
class ResNet101(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(in_channels, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            ResBlock(64, 256, 1, 1, num_layers=3),
            ResBlock(256, 512, 2, 1, num_layers=4),
            ResBlock(512, 1024, 1, 2, num_layers=23),
            ResBlock(1024, 2048, 1, 4, num_layers=3)
        )

    def forward(self, x):
        return self.block(x)
        

# ASPP layer(FC6 ~ FC8 layer)
class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.rate6_block = nn.Sequential(
            conv_block(in_ch, in_ch, 3, 1, padding=6, dilation=6),
            conv_block(in_ch, in_ch, 1, 1, 0),
            conv_block(in_ch, out_ch, 1, 1, 0)
        )
        self.rate12_block = nn.Sequential(
            conv_block(in_ch, in_ch, 3, 1, padding=12, dilation=12),
            conv_block(in_ch, in_ch, 1, 1, 0),
            conv_block(in_ch, out_ch, 1, 1, 0)
        )
        self.rate18_block = nn.Sequential(
            conv_block(in_ch, in_ch, 3, 1, padding=18, dilation=18),
            conv_block(in_ch, in_ch, 1, 1, 0),
            conv_block(in_ch, out_ch, 1, 1, 0)        
        )
        self.rate24_block = nn.Sequential(
            conv_block(in_ch, in_ch, 3, 1, padding=24, dilation=24),
            conv_block(in_ch, in_ch, 1, 1, 0),
            conv_block(in_ch, out_ch, 1, 1, 0)            
        )
        
    def forward(self, x):
        out1 = self.rate6_block(x)
        out2 = self.rate12_block(x)
        out3 = self.rate18_block(x)
        out4 = self.rate24_block(x)
        out = out1 + out2 + out3 + out4
        return out
        

# DeepLabV2 model        
class DeepLabV2(BaseModel):
    def __init__(self, in_channels, num_classes=len(constants.CLASSES)):
        super().__init__()
        self.backbone = ResNet101(in_channels)
        self.aspp = AtrousSpatialPyramidPooling(2048, num_classes)
        
    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])
        
        backbone_out = self.backbone(x)
        aspp_out = self.aspp(backbone_out)
        
        out = F.interpolate(
            aspp_out, size=upsample_size, mode="bilinear", align_corners=True
        )
        return out
    