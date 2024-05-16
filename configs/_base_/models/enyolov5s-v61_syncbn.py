
# ==================================== model parameters ===========================================
num_classes = 4 # should be modified due to the dataset
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# Strides of multi-scale prior box
strides = [8, 16, 32]
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config
# Basic size of multi-scale prior box
anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
prior_match_thr = 4.  # Priori box matching threshold
# The obj loss weights of the three output layers
obj_level_weights = [4., 1., 0.4]

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

model = dict(
    type='enyolo.EnYOLO',
    data_preprocessor=dict(
        # type='YOLOv5DetDataPreprocessor',
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    res_data_preprocessor=dict(
        type='enyolo.PairedResDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmyolo.YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        out_indices=(0, 1, 2, 3, 4), # different from the original setting
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='mmyolo.YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='mmyolo.YOLOv5Head',
        head_module=dict(
            type='mmyolo.YOLOv5HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        bbox_coder=dict(type='mmyolo.YOLOv5BBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight *
            (num_classes / num_classes * 3 / num_det_layers)),
        loss_bbox=dict(
            type='mmyolo.IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight *
            ((640 / 640)**2 * 3 / num_det_layers)),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights),
    decode_head=dict(
        type='enyolo.DeYOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        loss_l1 = dict(type='L1Loss', loss_weight=10.),
        act_cfg=dict(type='ReLU', inplace=True)),
    test_cfg=model_test_cfg)