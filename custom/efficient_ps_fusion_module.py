import math

import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.cnn import kaiming_init

from torch import nn

from custom.camera_pspnet.camera_pspnet import CameraPSPNet
from custom.efficientNet import geffnet
from custom.efficientNet.geffnet.efficientnet_builder import InvertedResidual
from custom.fusion_blocks import FusedInvBlock, make_inv_block, FusedResBlock
from custom.mmdet.models import build_detector, EfficientPSSemanticHead, TWOWAYFPN
from inplace_abn import ABN
from inplace_abn import InPlaceABN, InPlaceABNSync

from custom.mmdet.ops import build_norm_layer

class LidarRescale(nn.Module):
    def __init__(self):
        super(LidarRescale, self).__init__()

    def forward(self, input: torch.Tensor, sensor_overlap: torch.Tensor, _scale_h=1, _scale_w=1):
        '''

        :param input: projection indices or range image
        :param sensor_overlap: overlap idx 0-44, 0-484 or 0-44, 897-13..
        :param _scale_w: scale --> for now 8 later different scales
        :param _scale_h: scale --> for now 8 later different scales
        :return:
        '''
        b, _, h, w = sensor_overlap.shape

        _sensor_overlap = sensor_overlap.clone()

        if _scale_w != 1 or _scale_h != 1:
            h = math.ceil(h / _scale_h)
            w = math.ceil(w / _scale_w)  # --> 60 --> b,2,44,484  <- b,c,45,121

            _sensor_overlap[:, 0] = _sensor_overlap[:, 0] / _scale_h
            _sensor_overlap[:, 1] = _sensor_overlap[:, 1] / _scale_w

            # TODO interpolation method -> nearest <-, bilinear, float?
            _sensor_overlap = F.interpolate(_sensor_overlap.float(), size=(h, w), mode="nearest").long()

        #
        mask = (_sensor_overlap[:, 0] >= 0) & (_sensor_overlap[:, 0] < input.shape[2]) & \
               (_sensor_overlap[:, 1] >= 0) & (_sensor_overlap[:, 1] < input.shape[3])

        return cropped_batched_indexing(input, _sensor_overlap, mask)


class ImageToLidarScale(nn.Module):
    def __init__(self):
        super(ImageToLidarScale, self).__init__()

    def forward(self, rv: torch.Tensor, transformed_img_features: torch.Tensor):
        _, _, h, w = rv.shape

        transformed_img_features = F.interpolate(transformed_img_features.float(), size=(h, w))

        return transformed_img_features


class ImageRescale(nn.Module):
    def __init__(self, scale):
        '''
        returns image features projected into range view

        '''
        super(ImageRescale, self).__init__()
        scale = torch.tensor(scale)
        self._scale = scale[None, :, None, None]

    def forward(self, image_features: torch.Tensor, c2l_proj_ind: torch.Tensor):
        b, c, h, w = image_features.shape
        self._scale = self._scale.type(image_features.dtype).to(c2l_proj_ind.device)

        c2l_proj_ind = torch.round(c2l_proj_ind.float() / self._scale).long()

        mask = (c2l_proj_ind[:, 0] >= 0) & (c2l_proj_ind[:, 0] < h) & \
               (c2l_proj_ind[:, 1] >= 0) & (c2l_proj_ind[:, 1] < w)

        return cropped_batched_indexing(image_features, c2l_proj_ind, mask)


class EffPSFusionNet(nn.Module):
    def __init__(self,
                 config_model,
                 classes: int = 19,
                 freeze_pretrained_lidar: bool = False,
                 freeze_pretrained_camera: bool = False):
        super(EffPSFusionNet, self).__init__()

        self.freeze_pretrained_lidar = freeze_pretrained_lidar
        self.freeze_pretrained_camera = freeze_pretrained_camera

        self.lidar_backbone = build_detector(config_model.lidar)
        if freeze_pretrained_lidar:
            for parameter in self.lidar_backbone.parameters():
                parameter.requires_grad = False
            self.lidar_backbone.eval()

        # camera_backbone
        self.camera_backbone = CameraPSPNet(layers=101, use_ppm=True)
        if freeze_pretrained_camera:
            for parameter in self.camera_backbone.parameters():
                parameter.requires_grad = False
            self.camera_backbone.eval()

        self.neck = TWOWAYFPN(
            in_channels=[36, 60, 168],
            out_channels=192,
            norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01,
                          requires_grad=True),
            act_cfg=None,
            num_outs=3
        )

        self.neck.init_weights()

        self.efficient_semantic_head = EfficientPSSemanticHead(
            in_channels=192,
            conv_out_channels=96,
            num_classes=classes,
            num_in_bins=3,
            ohem=0.25,
            norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01, requires_grad=True),
            act_cfg=None
        )

        self.conv_logits = nn.Conv2d(432, classes, 1)
        self.init_weights()

        # fusion and feature reduction blocks

        self._cam_inv_512_24 = make_inv_block(512, 24)
        self._cam_inv_1024_40 = make_inv_block(1024, 40)
        self._cam_inv_2048_112 = make_inv_block(2048, 112)

        self._cam_inv_4096_192 = make_inv_block(4096, 192)

        self._fusion_block_48_36 = FusedResBlock(in_channels=48, fused_features_out=36)
        self._fusion_block_80_60 = FusedResBlock(in_channels=80, fused_features_out=60)
        self._fusion_block_224_168 = FusedResBlock(in_channels=224, fused_features_out=168)
        self._fusion_block_384_288 = FusedResBlock(in_channels=384, fused_features_out=288)

        self._fusion_block_576_432 = FusedResBlock(in_channels=576, fused_features_out=432)

        self._rv_rescale_transform = LidarRescale()
        self._rematch_lidar = ImageToLidarScale()

        self._img_rescale_transform_8 = ImageRescale(scale=(8, 8))

    def init_weights(self):
        kaiming_init(self.conv_logits)

    def forward(self, input: InputContainer):
        # camera image
        rgb = input.sensor(SensorType.Camera_0).data(DataType.RGB_IMAGE)
        # rv image
        rv = input.sensor(SensorType.LIDAR_0).data(DataType.RANGE_IMAGE)
        # indices for rv --> rgb image mapping
        proj_indices = input.sensor(SensorType.LIDAR_0).data(DataType.SENSOR_PROJECTION)
        # actual sensor overlap
        sensor_overlap = input.sensor(SensorType.LIDAR_0).data(DataType.SENSOR_OVERLAP)

        lidar_eff_out, lidar_out = self.lidar_backbone(rv)

        h, w = rv.shape[-2:]
        lidar_1_1 = F.interpolate(lidar_eff_out[0], size=(h, w), mode='bilinear', align_corners=False)
        lidar_1_4 = F.interpolate(lidar_eff_out[1], size=(h, int(w // 4)), mode='bilinear', align_corners=False)
        lidar_2_8 = F.interpolate(lidar_eff_out[2], size=(int(h // 2), int(w // 8)), mode='bilinear', align_corners=False)

        lidar_out = F.interpolate(lidar_out, size=(h, w), mode='bilinear', align_corners=False)

        x2, x_tmp, x_out, x_ppm = self.camera_backbone(rgb)
        # print(x2.shape, x_tmp.shape, x_out.shape)
        # for training use rescale
        if self.training:
            # crop to sensor overlap, respect scales
            proj_indices = self._rv_rescale_transform(proj_indices, sensor_overlap)

            lidar_1_1 = self._rv_rescale_transform(lidar_1_1, sensor_overlap, 1, 1)
            lidar_1_4 = self._rv_rescale_transform(lidar_1_4, sensor_overlap, 1, 4)
            lidar_2_8 = self._rv_rescale_transform(lidar_2_8, sensor_overlap, 2, 8)

            lidar_out = self._rv_rescale_transform(lidar_out, sensor_overlap, 1, 1)
        else:
            sensor_overlap_ = sensor_overlap[0]

            proj_indices = proj_indices[..., sensor_overlap_[0, 0]:sensor_overlap_[0, 1],
                           sensor_overlap_[1, 0]:sensor_overlap_[1, 1]].clone()

            w_left_0 = sensor_overlap_[1, 0]  # // 2 // 4
            w_right0 = sensor_overlap_[1, 1]  # // 2 // 4
            lidar_1_1 = lidar_1_1[..., sensor_overlap_[0, 0]:sensor_overlap_[0, 1], w_left_0:w_right0].clone()

            w_bottom_1 = sensor_overlap_[0, 0]  # // 2
            w_top_1 = sensor_overlap_[0, 1]  # // 2
            w_left_1 = sensor_overlap_[1, 0] // 4  # // 8
            w_right1 = sensor_overlap_[1, 1] // 4  # // 8
            lidar_1_4 = lidar_1_4[..., w_bottom_1:w_top_1, w_left_1:w_right1].clone()

            w_bottom_2 = sensor_overlap_[0, 0] // 2
            w_top_2 = sensor_overlap_[0, 1] // 2
            w_left_2 = sensor_overlap_[1, 0] // 8
            w_right_2 = sensor_overlap_[1, 1] // 8

            lidar_2_8 = lidar_2_8[..., w_bottom_2:w_top_2, w_left_2:w_right_2].clone()
            lidar_out = lidar_out[..., sensor_overlap_[0, 0]:sensor_overlap_[0, 1], w_left_0:w_right0].clone()

        ### neck fusion ######################################################################

        img_feas_trans_x2 = self._img_rescale_transform_8(x2, proj_indices)
        img_feas_trans_x3 = self._img_rescale_transform_8(x_tmp, proj_indices)
        x_out = self._img_rescale_transform_8(x_out, proj_indices)
        x_ppm = self._img_rescale_transform_8(x_ppm, proj_indices)

        img_feas_trans_x2_1_1 = self._rematch_lidar(lidar_1_1, self._cam_inv_512_24(img_feas_trans_x2))
        img_feas_trans_x3_1_4 = self._rematch_lidar(lidar_1_4, self._cam_inv_1024_40(img_feas_trans_x3))
        img_feas_trans_x_out_2_8 = self._rematch_lidar(lidar_2_8, self._cam_inv_2048_112(x_out))

        fused_feas_1 = self._fusion_block_48_36(lidar_1_1, img_feas_trans_x2_1_1)
        fused_feas_2 = self._fusion_block_80_60(lidar_1_4, img_feas_trans_x3_1_4)
        fused_feas_3 = self._fusion_block_224_168(lidar_2_8, img_feas_trans_x_out_2_8)

        fused_features = self.neck([fused_feas_1, fused_feas_2, fused_feas_3])
        fused_features = self.efficient_semantic_head(fused_features)

        # head fusion #####################################################

        # cam_out = self._img_rescale_transform_8(cam_out, proj_indices)
        # reduce image features
        cam_out = self._rematch_lidar(lidar_out, self._cam_inv_4096_192(x_ppm))

        fused_feas_head = self._fusion_block_384_288(lidar_out, cam_out)

        # fusion of semantic head output of fused head and pretrained models after fusion

        x_fused = self._fusion_block_576_432(fused_feas_head, fused_features)

        x_fused = self.conv_logits(x_fused)

        return x_fused

    def train(self, mode: bool = ...):
        super().train(mode)
        if self.freeze_pretrained_lidar:
            self.lidar_backbone.eval()

        if self.freeze_pretrained_camera:
            self.camera_backbone.eval()

    def eval(self):
        super().train(False)

    def get_camera_backbone_params(self):
        return self.camera_backbone.parameters()

    def get_lidar_backbone_params(self):
        return self.lidar_backbone.parameters()

    def get_fusion_params(self):
        params = [param for name, param in self.named_parameters() if 'lidar_backbone' not in name
                  and 'camera_backbone' not in name]
        return iter(params)