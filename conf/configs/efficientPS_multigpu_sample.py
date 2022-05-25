# model settings
model = dict(
    lidar=dict(type='EfficientPS',
               pretrained=False,
               backbone=dict(
                   in_channels=6,
                   type='tf_efficientnet_b1',
                   act_cfg=dict(type="Identity"),
                   norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01,
                                 requires_grad=True),
                   style='pytorch'),
               neck=dict(
                   type='TWOWAYFPN',
                   in_channels=[24, 40, 112],  # b0[24, 40, 112, 1280], #b4[32, 56, 160, 1792],
                   out_channels=128,
                   norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01,
                                 requires_grad=True),
                   act_cfg=None,
                   num_outs=3),
               semantic_head=dict(
                   type='EfficientPSSemanticHead',
                   in_channels=128,
                   conv_out_channels=64,
                   num_classes=19,
                   num_in_bins=3,
                   ohem=0.25,
                   norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01,
                                 requires_grad=True),
                   act_cfg=None)),
    camera=dict(type='EfficientPS',
                pretrained=False,
                backbone=dict(
                    in_channels=3,
                    type='tf_efficientnet_b5',
                    act_cfg=dict(type="Identity"),
                    norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01,
                                  requires_grad=True),
                    style='pytorch'),
                neck=dict(
                    type='TWOWAYFPN',
                    in_channels=[40, 64, 176, 2048],  # b0[24, 40, 112, 1280], #b4[32, 56, 160, 1792],
                    out_channels=256,
                    norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01,
                                  requires_grad=True),
                    act_cfg=None,
                    num_outs=4),
                semantic_head=dict(
                    type='EfficientPSSemanticHead',
                    in_channels=256,
                    conv_out_channels=128,
                    num_classes=19,
                    num_in_bins=4,
                    ohem=0.25,
                    norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01,
                                  requires_grad=True),
                    act_cfg=None)
                )
)
