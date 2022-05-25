from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import geffnet

from custom.efficientNet.geffnet.activations.activations import Sigmoid
from .. import builder
from ..registry import EFFICIENTPS
from .base import BaseDetector
from custom.mmdet.ops.norm import norm_cfg

import time
import cv2
import numpy as np

from ...ops import ConvModule


class FeatureFusion(nn.Module):
    def __init__(self, norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu',
                                     activation_param=0.01, requires_grad=True)):
        super(FeatureFusion, self).__init__()
        self.in_channels = [296, 320, 432, 2304]
        self.convs = nn.ModuleList()
        for i in range(0, 4):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=None,
                    act_cfg=None,
                    norm_cfg=norm_cfg))

        self.sigmoid = Sigmoid()
        self.conv_1x1 = ConvModule(
            256,
            256,
            1,
            conv_cfg=None,
            act_cfg=None,
            norm_cfg=norm_cfg)

    def forward(self, p_s, range_s):

        '''
        :param p_s: features from 2 way fpn
        :param range_s: features from REN
        :return: fused features
        '''
        # torch.autograd.set_detect_anomaly(True)
        p_s = list(p_s)
        range_s = list(range_s)
        fused_feats = []
        print(len(p_s), len(range_s), len(self.convs))
        assert len(p_s) == len(range_s) == len(self.convs)

        for i in range(len(p_s)):
            # REN and FPN concat at each scale
            fused_feats.append(torch.cat((p_s[i], range_s[i]), dim=1))
        print(len(fused_feats))
        for i in range(len(p_s)):
            print(fused_feats[i].shape)
            z = self.convs[i](fused_feats[i])
            print(z.shape)
            z = self.conv_1x1(z)
            fused_feats[i] = self.sigmoid(z)
        return fused_feats


@EFFICIENTPS.register_module
class EfficientPS(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 semantic_head=None,
                 shared_head=None,
                 pretrained=None):
        assert backbone is not None

        super(EfficientPS, self).__init__()

        self.eff_backbone_flag = False if 'efficient' not in backbone['type'] else True

        print(backbone)

        if self.eff_backbone_flag == False:
            self.backbone = builder.build_backbone(backbone)
        else:
            # type = tf_efficientnet_b5
            # scaling coefficient 1.6 2.2 456
            self.backbone = geffnet.create_model(backbone['type'],
                                                 pretrained=True if pretrained is not None else False,
                                                 se=False,
                                                 # type = tf_efficientnet_b5
                                                 act_layer=backbone['act_cfg']['type'],
                                                 norm_layer=norm_cfg[backbone['norm_cfg']['type']][1],
                                                 in_channels=backbone['in_channels'])

        print('num_outs', neck['num_outs'])
        self._num_out = neck['num_outs']
        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if semantic_head is not None:
            self.semantic_head = builder.build_head(semantic_head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if self.eff_backbone_flag == False:
            self.backbone.init_weights(pretrained=pretrained)

        self.neck.init_weights()

        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        # self.semantic_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        y = self.neck(x)
        return x, y

    def forward_train(self, img):
        x, y = self.extract_feat(img)
        semantic = self.semantic_head(y[:self._num_out])
        return x, semantic
