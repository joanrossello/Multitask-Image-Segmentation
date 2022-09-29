"""
Author: Group work
Date: 2021/12/26
Description: baseline model for segementation task - U-net
"""

# import packages
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv -> bn -> relu) *2
    """
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    """Downsample using maxpool with double conv
    """
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):
    """Upsample with double conv
    """
    def __init__(self, in_channel, out_channel):
        super(UpSample,self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x, prev_x):
        x = self.up(x)
        x = torch.cat([x, prev_x], axis=1)
        return self.conv(x)
    
class UNet(nn.Module):
    """UNet baseline model
    """
    def __init__(self):
        super(UNet,self).__init__()
        self.in_conv = DoubleConv(3,64)
        self.d1 = DownSample(64,128)
        self.d2 = DownSample(128,256)
        self.d3 = DownSample(256,512)
        self.d4 = DownSample(512,1024)
        
        self.u1 = UpSample(1024, 512)
        self.u2 = UpSample(512, 256)
        self.u3 = UpSample(256, 128)
        self.u4 = UpSample(128, 64)
        
        self.out_conv = nn.Conv2d(64,1,3,1,1)
        self.Th = nn.Sigmoid()
    
    def forward(self,x):
        x1 = self.in_conv(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        out = self.out_conv(x)
        return out

if __name__ == '__main__':
    x=torch.randn(1,3,256,256)
    net=UNet()
    pred = net(x)
    print(pred.shape)
    
    