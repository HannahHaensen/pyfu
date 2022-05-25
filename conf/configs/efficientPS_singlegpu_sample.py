# model settings
model = dict(
    type='EfficientPS',
    pretrained=True,
    backbone=dict(
        type='tf_efficientnet_b5',
        act_cfg = dict(type="Identity"),  
        norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='TWOWAYFPN',
        in_channels=[40, 64, 176, 2048], #b0[24, 40, 112, 1280], #b4[32, 56, 160, 1792],
        out_channels=256,
        norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
        act_cfg=None,
        num_outs=4),
    semantic_head=dict(
        type='EfficientPSSemanticHead',
        in_channels=256,
        conv_out_channels=128,
        num_classes=19,
        ignore_label=255,
        loss_weight=1.0,
        ohem=0.25,
        norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
        act_cfg=None))