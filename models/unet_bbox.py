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


class BoundingBoxes(nn.Module):

    def __init__(self):
        super(BoundingBoxes, self).__init__()
        self.fbb1 = nn.Linear(1024 * 16 * 16, 128)
        self.fbb2 = nn.Linear(128, 64)
        self.fbb3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fbb1(x))
        x = F.relu(self.fbb2(x))
        return self.fbb3(x)


class UNET_BBOX(nn.Module):

    def __init__(self, in_ch):
        super(UNET_BBOX, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = DoubleConv(in_ch, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.bbox = BoundingBoxes()

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

        # bounding boxes mdoel
        bboxes = self.bbox(x)

        return bboxes

if __name__ == '__main__':
    model = UNET_BBOX(in_ch=3, out_ch=1)
    x = torch.randn((1, 3, 256, 256))
    pred = model(x)
    print(pred.shape)
    print(pred)