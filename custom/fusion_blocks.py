import torch
from torch import nn

from custom.efficientNet import geffnet
from custom.efficientNet.geffnet.efficientnet_builder import InvertedResidual
from inplace_abn import ABN
from inplace_abn import InPlaceABN, InPlaceABNSync


def make_inv_block(in_channels, out_channels, k_size=1):
    return InvertedResidual(in_chs=in_channels,
                            out_chs=out_channels,
                            dw_kernel_size=k_size,
                            stride=1,
                            pad_type='same',
                            act_layer=geffnet.activations.activations.Identity,
                            noskip=False,
                            exp_kernel_size=k_size,
                            pw_kernel_size=k_size,
                            se_ratio=None,
                            se_kwargs=None,
                            norm_layer=InPlaceABNSync,
                            norm_kwargs={'eps': 0.001},
                            conv_kwargs=None,
                            drop_connect_rate=0.0
                            )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = InPlaceABNSync(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = InPlaceABNSync(planes)
        self.relu = nn.LeakyReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = InPlaceABNSync(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = InPlaceABNSync(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = InPlaceABNSync(planes * self.expansion)
        self.relu = nn.LeakyReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class FusedResBlock(nn.Module):
    def __init__(self, in_channels=128, fused_features_out=192, blocks=1):
        super(FusedResBlock, self).__init__()
        self.sensor_fusion = Bottleneck(in_channels, in_channels // 2, stride=1)
        self.block_fusion = make_layer(BasicBlock, in_channels, fused_features_out, blocks, stride=1)

    def forward(self, lidar_feats, image_feats):
        x = torch.cat((lidar_feats, image_feats), dim=1)
        x = self.sensor_fusion(x)
        x = self.block_fusion(x)
        return x


class FusedInvBlock(nn.Module):
    def __init__(self, in_channels=128, fused_features_out=192):
        super(FusedInvBlock, self).__init__()
        self.sensor_fusion = make_inv_block(in_channels, in_channels // 2)
        self.block_fusion = make_inv_block(in_channels // 2, fused_features_out)

    def forward(self, lidar_feats, image_feats):
        x = torch.cat((lidar_feats, image_feats), dim=1)
        x = self.sensor_fusion(x)
        x = self.block_fusion(x)
        return x
