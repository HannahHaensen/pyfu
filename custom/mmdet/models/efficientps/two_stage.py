import torch.nn as nn
import geffnet
import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import EFFICIENTPS
from ...ops.norm import norm_cfg


@EFFICIENTPS.register_module
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()

        self.eff_backbone_flag = False if 'efficient' not in backbone['type'] else True

        if self.eff_backbone_flag == False:
            self.backbone = builder.build_backbone(backbone)
        else:
            self.backbone = geffnet.create_model(backbone['type'], 
                                                 pretrained=True if pretrained is not None else False,
                                                 se=False, 
                                                 act_layer=backbone['act_cfg']['type'],
                                                 norm_layer=norm_cfg[backbone['norm_cfg']['type']][1]) 

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        if self.eff_backbone_flag == False:
            self.backbone.init_weights(pretrained=pretrained)

        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, input):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled

        Returns:
            extracted features
        """
        x = self.extract_feat(input)
        return x