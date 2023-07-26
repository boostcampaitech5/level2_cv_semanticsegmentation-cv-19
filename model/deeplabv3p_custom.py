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


# Depthwise Separable Convolution 연산 
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation=1):
        super().__init__()
        if dilation > kernel_size//2:
            padding = dilation
        else:
            padding = kernel_size//2
            
        self.depthwise_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding,
            dilation=dilation, groups=in_ch, bias=False
        )
        self.pointwise_conv = nn.Conv2d(
            in_ch, out_ch, 1, 1, bias=False
        )
        self.bn = nn.BatchNorm2d(in_ch)
        
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.bn(out)
        out = self.pointwise_conv(out)
        return out


# XceptionBlock for Xception
class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super().__init__()
        if in_ch != out_ch or stride !=1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else: 
            self.skip = None
        
        if exit_flow:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, in_ch, 3, 1, dilation),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch) 
            ]
        else:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),            
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),            
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch)                
            ]
   
        if not use_1st_relu: 
            block = block[1:]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        output = self.block(x)
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        x = output + skip
        return x
    

# Modified Xception backbone(Encoder)
class Xception(nn.Module):
    def __init__(self, in_channels):
        super(Xception, self).__init__()        
        self.entry_block_1 = nn.Sequential(
            conv_block(in_channels, 32, 3, 2, 1),
            conv_block(32, 64, 3, 1, 1, relu=False),
            XceptionBlock(64, 128, 2, 1, use_1st_relu=False)
        )
        self.relu = nn.ReLU()
        self.entry_block_2 = nn.Sequential(
            XceptionBlock(128, 256, 2, 1),
            XceptionBlock(256, 728, 2, 1)
        )
        
        middle_block = [XceptionBlock(728, 728, 1, 1) for _ in range(16)]
        self.middle_block = nn.Sequential(*middle_block)
        
        self.exit_block = nn.Sequential(
            XceptionBlock(728, 1024, 1, 1, exit_flow=True),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1024, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 2048, 3, 1, 2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )
            
    def forward(self, x):
        out = self.entry_block_1(x)
        features = out
        out = self.entry_block_2(out)
        out = self.middle_block(out)
        out = self.exit_block(out)
        return out, features
    

# ASPP + AdaptiveAvgPooling layer(Encoder)
class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block1 = conv_block(in_ch, 256, 1, 1, 0, 1)
        self.block2 = conv_block(in_ch, 256, 3, 1, 6, 6)
        self.block3 = conv_block(in_ch, 256, 3, 1, 12, 12)
        self.block4 = conv_block(in_ch, 256, 3, 1, 18, 18)
        self.block5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv = conv_block(256*5, 256, 1, 1, 0)
         
    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])
        
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out5 = self.block5(x)
        out5 = F.interpolate(
            out5, size=upsample_size, mode="bilinear", align_corners=True
        )
        
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.conv(out)
        return out


# Decoder
class Decoder(nn.Module):
    def __init__(self, num_classes=len(constants.CLASSES)):
        super().__init__()
        self.block1 = conv_block(128, 48, 1, 1, 0)
        self.block2 = nn.Sequential(
            conv_block(48+256, 256, 3, 1, 1),
            conv_block(256, 256, 3, 1, 1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x, features):
        features = self.block1(features)
        feature_size = (features.shape[-1], features.shape[-2])
        
        out = F.interpolate(x, size=feature_size, mode="bilinear", align_corners=True)
        out = torch.cat((features, out), dim=1)
        out = self.block2(out)
        return out


# DeepLabV3+ model(Encoder + Decoder)    
class DeepLabV3p(BaseModel):
    def __init__(self, in_channels=3, num_classes=len(constants.CLASSES)):
        super().__init__()
        self.backbone = Xception(in_channels)
        self.aspp = AtrousSpatialPyramidPooling(2048)
        self.decoder = Decoder(num_classes)
        
    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])

        backbone_out, features = self.backbone(x)
        aspp_out = self.aspp(backbone_out)
        
        out = self.decoder(aspp_out, features)
        out = F.interpolate(
            out, size=upsample_size, mode="bilinear", align_corners=True
        )
        return out
    