# ========================Possible modified parameters========================
# -------------------------------- data related ------------------------------
img_scale = (640, 640) # width, height
dataset_type = 'mmyolo.YOLOv5CocoDataset'
data_root = './data/duo/'  # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/instances_train.json'
train_data_prefix = 'images/train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/instances_test.json'
val_data_prefix = 'images/test/'  # Prefix of val image path
# detection classes
class_name = ('holothurian', 'echinus', 'scallop', 'starfish')
# num_classes = len(class_name)  # Number of classes for classification
metainfo = dict(classes=class_name,
                palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 0
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2
# persistent_workers must be False if num_workers is 0
persistent_workers = True if train_num_workers > 0 else False
# ------------------------------ data for restoration --------------------------
res_dataset_type = 'enyolo.PairedImagesDataset'
res_data_root = './data/syrea'
res_train_batch_size_per_gpu = 8
# ------------------------------ data for detection in mutual-learning & domain-adapation stages ---
det_dataset_type = dataset_type
det_data_root = data_root
det_train_batch_size_per_gpu = res_train_batch_size_per_gpu


# ------------------------------ train dataloader for detection ----------------
affine_scale = 0.5
max_aspect_ratio = 100
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]
mosaic_affine_pipeline = [
    dict(
        type='mmyolo.Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='mmyolo.YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]
train_pipeline = [
    *pre_transform, *mosaic_affine_pipeline,
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='mmyolo.YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))
# ------------------------- train dataloader for restoration -----------------------------------
pre_transform_res = [
    dict(type='enyolo.LoadPairedImagesFromFile')
]
train_pipeline_res = [
    *pre_transform_res,
    dict(type='enyolo.ResizePairedImage',
         scale=img_scale, keep_ratio=True),
    dict(type='enyolo.RandomFlipPairedImage',
         prob=0.5),
    dict(type='enyolo.RandomCropPairedImage',
         crop_size=(0.6, 0.6), crop_type='relative_range'),
    dict(type='enyolo.PackResInputs')
]
train_dataloader_res = dict(
    batch_size=res_train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=res_dataset_type,
        pipeline=train_pipeline_res,
        ann_file='train_infos.json',
        data_root=res_data_root,
        data_prefix=dict(input='degraded/', target='clear/'),
        img_suffix='.png',
        test_mode=False),
    collate_fn=dict(type='paired_image_collate')
)
# =============================== Dataloader for detection in mutual-learning stage ====================
pre_transform_det = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]
train_pipeline_det = [
    *pre_transform_det, # no mosaic_affine_pipeline and random colorazation
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='mmyolo.YOLOv5HSVRandomAug'),
    dict(type='mmyolo.YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='mmyolo.LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='mmyolo.PPYOLOERandomCrop', aspect_ratio=[0.5, 1.0], thresholds=[0.2]),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader_det = dict(
    batch_size=det_train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline_det)
)

# ----------------------------- test dataloader ------------------------------------
batch_shapes_cfg = dict(
    type='mmyolo.BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='mmyolo.YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='mmyolo.LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

# -------------------------- test dataloader for restoration ----------------------------------
res_test_dataset_type = 'enyolo.TestImagesDataset'
res_test_data_root = './data/duo/images/test/'
test_pipeline_res = [
    dict(type='enyolo.LoadTestImagesFromFile'),
    dict(type='enyolo.ResizeTestImage',
         scale=img_scale, keep_ratio=True),
    dict(type='enyolo.PackResInputs',
         meta_keys=('file_name', 'img_shape', 'ori_shape', 'scale_factor'))
]
test_dataloader_res = dict(
    batch_size=1,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=res_test_dataset_type,
        pipeline=test_pipeline_res,
        data_root=res_test_data_root,
        img_suffix='.png',
        test_mode=False),
    collate_fn=dict(type='test_image_collate')
)