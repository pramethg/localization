import torch
import numpy as np
import torch.nn as nn
from torchviz import make_dot
import torch.nn.functional as F
from dsntnn.dsntnn import flat_softmax, dsnt

class NyquistConvolution(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        super(NyquistConvolution, self).__init__()
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
    @staticmethod
    def conv3x3(in_channels, out_channels, stride):
        return nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels,
                stride = stride,
                kernel_size = (3, 3),
                padding = 1,
                bias = False
            )
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualConvModule, self).__init__()
        self.stride = stride
        self.conv1 = nn.Sequential(
            self.conv3x3(in_channels = in_channels, out_channels = out_channels, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
            )
        self.conv2 = nn.Sequential(
            self.conv3x3(in_channels = out_channels, out_channels = out_channels, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
            )
        self.strideconv = nn.Sequential(
            self.conv3x3(in_channels = out_channels, out_channels = out_channels, stride = stride),
            nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        out = self.strideconv(out)
        if self.stride == 1:
            out += x
        return F.relu(out)

class UpsamplingModule(nn.Module):
    @staticmethod
    def conv3x3(in_channels, out_channels, stride):
        return nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                stride = stride,
                kernel_size = (3, 3),
                padding = 1,
                bias = False
            )
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = (2, 2),
                stride = 2
            ),
            self.conv3x3(in_channels = out_channels, out_channels = out_channels, stride = 1)
        )
    def forward(self, x):
        return self.upsample(x)

class LocalizationNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, features = np.array([16, 32, 64, 128, 256, 256])):
        super(LocalizationNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skipconnections = []
        self.features = np.insert(features, 0, self.in_channels)
        self.highfovconv = nn.Conv2d(
                in_channels = features[-1], 
                out_channels = features[-1], 
                kernel_size = (5, 5), 
                stride = 1, 
                padding = (2, 2)
        )
        self.downsample = nn.ModuleList()
        self.downsample.append(ResidualConvModule(in_channels = self.features[0], out_channels = self.features[1], stride = 1))
        for idx in range(1, len(self.features) - 1):
            self.downsample.append(ResidualConvModule(in_channels = self.features[idx], out_channels = self.features[idx + 1], stride = 2))
        self.downsample.extend([
            self.highfovconv,
            ResidualConvModule(in_channels = self.features[-1], out_channels = self.features[-1], stride = 1),
            self.highfovconv,
        ])

        self.singleconv = nn.Conv2d(
                in_channels = self.features[1],
                out_channels = self.features[0],
                kernel_size = (1, 1),
                stride = 1,
                bias = False
        )
        self.upsample = nn.ModuleList()
        self.upsample.append(ResidualConvModule(in_channels = self.features[-1], out_channels = self.features[-1], stride = 1))
        for idx in range(1, 6):
            self.upsample.extend([
                ResidualConvModule(in_channels = self.features[-idx], out_channels = self.features[-idx], stride = 1),
                UpsamplingModule(in_channels = self.features[-idx], out_channels = self.features[-(idx+1)]),
            ])
        self.upsample.append(ResidualConvModule(in_channels = self.features[1], out_channels = self.features[1], stride = 1))
        self.upsample.append(self.singleconv)

        self.heatmaps = None
        self.coords = None

    def forward(self, x):
        x = NyquistConvolution(in_channels = self.in_channels, out_channels = self.in_channels)(x)
        for idx, downsample in enumerate(self.downsample):
            x = downsample(x)
            # print(x.shape)
            if idx in [1, 2, 3, 4, 6]:
                self.skipconnections.append(x)
        for idx, upsample in enumerate(self.upsample):
            if idx in list(range(2, 11, 2)):
                x = torch.add(x, self.skipconnections[-(idx//2)])
            x = upsample(x)
        self.heatmaps = flat_softmax(x)
        self.coords = dsnt(self.heatmaps)
        return self.coords, self.heatmaps, x

def test(save = False, make_dot = False):
    x = torch.randn((1, 1, 256, 1024))
    model = LocalizationNet(1, 1)
    pred = model(x)
    assert(pred[2].shape == torch.Size([1, 1, 256, 256]))
    print(pred[2].shape)
    if make_dot:
        make_dot(pred[2], params=dict(list(model.named_parameters()))).render("model", format="png")
    if save:
        torch.onnx.export(
            model = model,
            args = x,
            f = "./local-pa.onnx",
            input_names = ['Input Sensor Data'],
            output_names = ['Reconstructed PA Image'],
            verbose = True
        )

if __name__ == "__main__":
    test(save = False, make_dot = False)