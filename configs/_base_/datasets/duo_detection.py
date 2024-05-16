# dataset settings
dataset_type = 'CocoDataset'
data_root = './data/duo/'  # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/instances_train.json'
train_data_prefix = 'images/train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/instances_test.json'
val_data_prefix = 'images/test/'  # Prefix of val image path
class_name = ('holothurian', 'echinus', 'scallop', 'starfish')
num_classes = len(class_name)  # Number of classes for classification
metainfo = dict(classes=class_name,
                palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

img_scale = (640, 640)
crop_size = (256, 256)
train_batch_size_per_gpu = 12
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2
# persistent_workers must be False if num_workers is 0
persistent_workers = True if train_num_workers > 0 else False

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

train_pipeline = [
    *pre_transform,
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
    # dict(type='mmyolo.PPYOLOERandomCrop', aspect_ratio=[0.5, 1.0], thresholds=[0.2]),
    dict(type='mmdet.RandomCrop', crop_size=crop_size),
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
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='mmyolo.YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root+val_ann_file,
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
