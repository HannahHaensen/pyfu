import torch
from pytorchbase.nn_utility.container import InputContainer
from torch import nn
import torch.nn.functional as F

from custom.camera_pspnet.camera_resnet import resnet50, resnet101, resnet152


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class CameraPSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), zoom_factor=8, use_ppm=True, pretrained=False):
        super(CameraPSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm

        if layers == 50:
            resnet = resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = resnet101(pretrained=pretrained)
        else:
            resnet = resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
            resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)

    def forward(self, x: InputContainer):

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x_tmp = self.layer3(x2)
        x_out = self.layer4(x_tmp)
        if self.use_ppm:
            ppm = self.ppm(x_out)
            # print(x2.shape, x_tmp.shape, x_out.shape, ppm.shape)
            return x2, x_tmp, x_out, ppm
        return x2, x_tmp, x_out, x_out