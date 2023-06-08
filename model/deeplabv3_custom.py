import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            conv_block(in_ch, out_ch // 4, 1, 1, 0),
            conv_block(out_ch // 4, out_ch // 4, 3, stride, dilation, dilation),
            conv_block(out_ch // 4, out_ch, 1, 1, 0, relu=False),
        )
        self.downsample = nn.Sequential(conv_block(in_ch, out_ch, 1, stride, 0, 1, False)) if downsample else None

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
            block.append(Bottleneck(in_ch if i == 0 else out_ch, out_ch, stride if i == 0 else 1, dilation, True if i == 0 else False))

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
            ResBlock(1024, 2048, 1, 4, num_layers=3),
        )

    def forward(self, x):
        return self.block(x)


# ASPP + AdaptiveAvgPooling layer(FC6 ~ FC8 layer)
class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block1 = conv_block(in_ch, out_ch, 1, 1, padding=0)
        self.block2 = conv_block(in_ch, out_ch, 3, 1, padding=6, dilation=6)
        self.block3 = conv_block(in_ch, out_ch, 3, 1, padding=12, dilation=12)
        self.block4 = conv_block(in_ch, out_ch, 3, 1, padding=18, dilation=18)
        self.block5 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())

    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])

        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out5 = self.block5(x)
        out5 = F.interpolate(out5, size=upsample_size, mode="bilinear", align_corners=False)

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return out


# DeepLabV3 model
class DeepLabV3(BaseModel):
    def __init__(self, in_channels=3, num_classes=len(constants.CLASSES)):
        super().__init__()
        self.backbone = ResNet101(in_channels)
        self.aspp = AtrousSpatialPyramidPooling(2048, 256)
        self.conv1 = conv_block(256 * 5, 256, 1, 1, 0)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])

        backbone_out = self.backbone(x)
        aspp_out = self.aspp(backbone_out)
        out = self.conv1(aspp_out)
        out = self.conv2(out)

        out = F.interpolate(out, size=upsample_size, mode="bilinear", align_corners=True)
        return out
