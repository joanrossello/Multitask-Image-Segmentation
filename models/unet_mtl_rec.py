import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        '''self.conv = nn.Conv2d(1024, 2048, kernel_size=3,
                      padding=1, stride=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)'''
        self.fc1 = nn.Linear(1024 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        '''x1 = self.conv(x)
        x1 = self.pool(x1)'''
        x1 = torch.flatten(x, 1)
        x1 = F.elu(self.fc1(x1))
        x1 = F.elu(self.fc2(x1))
        return self.fc3(x1)


class UNET_MTL(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UNET_MTL, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = DoubleConv(in_ch, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.classify = Classifier()

        self.conv_up1_seg = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1_seg = DoubleConv(1024, 512)
        self.conv_up1_rec = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1_rec = DoubleConv(1024, 512)

        self.conv_up2_seg = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2_seg = DoubleConv(512, 256)
        self.conv_up2_rec = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2_rec = DoubleConv(512, 256)

        self.conv_up3_seg = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_seg = DoubleConv(256, 128)
        self.conv_up3_rec = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_rec = DoubleConv(256, 128)

        self.conv_up4_seg = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4_seg = DoubleConv(128, 64)
        self.conv_up4_rec = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4_rec = DoubleConv(128, 64)

        self.out_seg = nn.Conv2d(64, out_ch, kernel_size=1)
        self.out_rec = nn.Conv2d(64, 3, kernel_size=1)


    def crop(self, skip, x):
        H, W = x.shape[2], x.shape[3]
        return torchvision.transforms.CenterCrop([H, W])(skip)
        
    def forward(self, x):
        skips = []

        # encoder
        x = self.down1(x)
        skips.append(x)
        x = self.pool(x)

        x = self.down2(x)
        skips.append(x)
        x = self.pool(x)

        x = self.down3(x)
        skips.append(x)
        x = self.pool(x)

        x = self.down4(x)
        skips.append(x)
        x = self.pool(x)

        # bottom level
        x = self.bottleneck(x)

        # classification model
        label = self.classify(x)
        
        # decoder, segmentation
        x_seg = self.conv_up1_seg(x)
        skip = self.crop(skips[-1], x_seg)
        x_seg = self.up1_seg(torch.cat((skip, x_seg), dim=1))

        x_rec = self.conv_up1_rec(x)
        skip = self.crop(skips[-1], x_rec)
        x_rec = self.up1_rec(torch.cat((skip, x_rec), dim=1))

        x_seg = self.conv_up2_seg(x_seg)
        skip = self.crop(skips[-2], x_seg)
        x_seg = self.up2_seg(torch.cat((skip, x_seg), dim=1))

        x_rec = self.conv_up2_rec(x_rec)
        skip = self.crop(skips[-2], x_rec)
        x_rec = self.up2_rec(torch.cat((skip, x_rec), dim=1))

        x_seg = self.conv_up3_seg(x_seg)
        skip = self.crop(skips[-3], x_seg)
        x_seg = self.up3_seg(torch.cat((skip, x_seg), dim=1))

        x_rec = self.conv_up3_rec(x_rec)
        skip = self.crop(skips[-3], x_rec)
        x_rec = self.up3_rec(torch.cat((skip, x_rec), dim=1))

        x_seg = self.conv_up4_seg(x_seg)
        skip = self.crop(skips[-4], x_seg)
        x_seg = self.up4_seg(torch.cat((skip, x_seg), dim=1))

        x_rec = self.conv_up4_rec(x_rec)
        skip = self.crop(skips[-4], x_rec)
        x_rec = self.up4_rec(torch.cat((skip, x_rec), dim=1))

        return self.out_seg(x_seg), self.out_rec(x_rec), label


if __name__ == '__main__':
    model = UNET_MTL(in_ch=3, out_ch=1)
    x = torch.randn((1, 3, 256, 256))
    pred = model(x)
