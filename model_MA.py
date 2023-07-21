import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import cv2
from PIL import Image
from torchvision import transforms
import random
import torchvision.models as models
from torchvision.models.resnet import ResNet34_Weights
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

resnet = models.resnet34(weights=ResNet34_Weights["IMAGENET1K_V1"])
for param in resnet.parameters():
    param.requires_grad = False

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

class MiddleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(1,1), padding=(1,1))
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(1,1), padding=(1,1))
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.bn(x)

        return x

# basic block untuk decoder
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(1,1), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(1,1), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x

class Dec_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Up-Convolution
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # LeakyReLU
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)

        #dropout layer
        self.dropout = nn.Dropout()

        # Basic Block
        self.conv = BasicBlock(in_channels // 2 + out_channels, out_channels)

    def forward(self, inputs, skip):
        up_x = self.upconv(inputs)
        up_x = self.relu(up_x)
        up_x = self.bn(up_x)  
        skip = nn.functional.interpolate(skip, size=up_x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([up_x, skip], dim=1)
        x = self.dropout(x)
        x = self.conv(x)

        return x
    
class MergeLayer(nn.Module):
    def __init__(self):
        super(MergeLayer, self).__init__()

    def forward(self, x, skip):
        #x = nn.functional.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return x
    
class Unet_MA(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.encoder = Encoder()

        """ Middle Convolution """
        self.mc1 = MiddleConv(512, 1024)
        self.mc2 = MiddleConv(1024, 1024)

        """ Decoder """
        self.decoder1 = Dec_Block(1024, 512)
        self.decoder2 = Dec_Block(512, 256)
        self.decoder3 = Dec_Block(256, 128)
        self.decoder4 = Dec_Block(128, 64)

        """ Merge Layer """
        self.merge = MergeLayer()

        """ Segmentation Convolution """
        self.segmentation = nn.Conv2d(67, 1, kernel_size=1, padding=0)
        
    def forward(self, x):
        """ Encoder """
        x1, x2, x3, x4 = self.encoder(x)

        mc1 = self.mc1(x4)

        """ Decoder """
        d1 = self.decoder1(mc1, x4)
        d2 = self.decoder2(d1, x3)
        d3 = self.decoder3(d2, x2)
        d4 = self.decoder4(d3, x1)

        merged = self.merge(d4, x)
        
        out = self.segmentation(merged)
        
        return out