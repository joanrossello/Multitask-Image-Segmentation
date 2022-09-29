import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

# Network with classification and bounding boxes in the same branch, and then we separate on the last layer.

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

class Branch(nn.Module):

    def __init__(self):
        super(Branch, self).__init__()
        self.fc1 = nn.Linear(1024 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc1(x)

class BoundingBoxes(nn.Module):

    def __init__(self):
        super(BoundingBoxes, self).__init__()
        self.fbb1 = nn.Linear(64, 4)

    def forward(self, x):
        return self.fbb1(x)


class UNET_MTL(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UNET_MTL, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = DoubleConv(in_ch, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.branch = Branch()
        
        self.classify = Classifier()

        self.bbox = BoundingBoxes()

        self.conv_up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1 = DoubleConv(1024, 512)
        self.conv_up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = DoubleConv(512, 256)
        self.conv_up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(256, 128)
        self.conv_up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(128, 64)

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

        x = self.down4(x)
        skips.append(x)
        x = self.pool(x)

        # bottom level
        x = self.bottleneck(x)

        # Branch that then divides into classification and bbox
        b = self.branch(x)

        # classification model
        label = self.classify(b)

        # bounding boxes mdoel
        bboxes = self.bbox(b)
        
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

        x = self.conv_up4(x)
        skip = self.crop(skips[-4], x)
        x = self.up4(torch.cat((skip, x), dim=1))

        return self.out(x), label, bboxes


if __name__ == '__main__':
    model = UNET_MTL(in_ch=3, out_ch=1)
    x = torch.randn((1, 3, 256, 256))
    pred = model(x)
