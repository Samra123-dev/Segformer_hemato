# hemato_binary.py

dataset_type = 'HematoDataset'
data_root = '/media/iml/cv-lab/Datasets_B_cells/Hemato_Data'  # Update path if needed

# ✅ Binary segmentation class definitions
classes = ('Background', 'WBC')
palette = [
    [0, 0, 0],      # Background - black
    [255, 255, 255] # WBC - white
]

# Image normalization (ImageNet stats)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Crop size
crop_size = (512, 512)

# ✅ Training pipeline with light augmentations (safe for binary)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=20),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# ✅ Testing pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# ✅ Data paths for binary mask folders
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='masks_binary/train',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='masks_binary/val',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='masks_binary/test',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        test_mode=True
    )
)
