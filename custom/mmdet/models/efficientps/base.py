from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn

from custom.mmdet.core import auto_fp16, get_classes, tensor2imgs
from custom.mmdet.utils import print_log


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors"""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        # img = img.sensor(SensorType.LIDAR_0).data(DataType.RANGE_IMAGE)
        return self.forward_train(img, **kwargs)
