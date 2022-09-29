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
        self.fc1 = nn.Linear(512 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = torch.flatten(x, 1)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        return self.fc3(x1)


class UNET_MTL(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UNET_MTL, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = DoubleConv(in_ch, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        # TODO: CHECK NETWORK FOR CLASSIFICATION (DENSE?), AND POSSIBLY CREATE HEAD BBOX MODEL
        self.classify = Classifier()

        self.conv_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up1 = DoubleConv(512, 256)
        self.conv_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)
        self.conv_up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

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

        # bottom level
        x = self.bottleneck(x)

        # classification model
        label = self.classify(x)
        
        # decoder, segmentation
        x = self.conv_up1(x)
        skip = self.crop(skips[-1], x)
        x = self.up1(torch.cat((skip, x), dim=1))

        x = self.conv_up2(x)
        skip = self.crop(skips[-2], x)
        x = self.up2(torch.cat((skip, x), dim=1))

        x = self.conv_up3(x)
        skip = self.crop(skips[-3], x)
        x = self.up3(torch.cat((skip, x), dim=1))

        return self.out(x), label


if __name__ == '__main__':
    model = UNET_MTL(in_ch=3, out_ch=1)
    x = torch.randn((1, 3, 256, 256))
    pred = model(x)
