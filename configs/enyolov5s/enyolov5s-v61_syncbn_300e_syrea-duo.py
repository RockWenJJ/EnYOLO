_base_ = ['../_base_/datasets/syrea-duo_enyolo.py',
          '../_base_/models/enyolov5s-v61_syncbn.py',
          '../_base_/default_runtime.py',
          '../_base_/default_schedule_20e.py',
          '../_base_/det_p5_tta.py']

runner_type = 'Runner4EnYOLO'
# ============================= model related =========================================
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

# ============================ train dataloader setups =======================================
train_batch_size_per_gpu = 16
train_num_workers = 8
persistent_workers = True if train_num_workers > 0 else False
res_train_batch_size_per_gpu = 6
det_train_batch_size_per_gpu = res_train_batch_size_per_gpu

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
)

train_dataloader_res = dict(
    batch_size=res_train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
)

train_dataloader_det = dict(
    batch_size=det_train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers
)

burnin_epochs = 200
mutual_epochs = 260
max_epochs = 300


val_interval = 2
base_lr = 0.001
lr_factor = 0.01
weight_decay = 0.0005
milestones = [burnin_epochs, mutual_epochs]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop4EnYOLO',
    burnin_epochs=burnin_epochs,
    mutual_epochs=mutual_epochs,
    max_epochs=max_epochs,
    val_interval=val_interval)

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    # classwise=True,
    ann_file=_base_.data_root + _base_.val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator



test_res_cfg = dict(type='enyolo.TestResLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=milestones,
        gamma=0.1),
]
# param_scheduler = None

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50))


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=_base_.train_batch_size_per_gpu),
    constructor='mmyolo.YOLOv5OptimizerConstructor')

