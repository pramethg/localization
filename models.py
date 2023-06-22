import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class NyquistConvolution(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        self.nyquistconv = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = (7, 7),
                stride = (1, 2),
                dilation = (1, 4),
                padding = (3, 12),
                bias = False
            ),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2)
            )
        )
    def forward(self,x):
        return self.nyquistconv(x)

class ResidualConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualConvModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels,
                kernel_size = (3, 3),
                stride = 1,
                padding = 1,
                bias = True
                ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
        self.strideconv = nn.Sequential(
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = (3, 3),
                stride = stride,
                padding  = 1,
                bias = True
            ),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        out = self.conv(x)
        out = self.conv(out)
        out = self.strideconv(out)
        return nn.ReLU(x + out)

class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = (2, 2),
                stride = 2
            ),
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = (3, 3),
                stride = 1,
                padding = 1,
                bias = True
            )
        )
    def forward(self, x):
        return self.upsample(x)

class LocalizationNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, features = [16, 32, 64, 128, 256]):
        self.skipconnections = []
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()

def test():
    x = torch.randn((1, 1, 256, 1024))
    model = LocalizationNet(1, 1)
    pred = model(x)
    print(pred.shape, x.shape)
    assert pred.shape == x.shape

if __name__ == "__main__":
    pass