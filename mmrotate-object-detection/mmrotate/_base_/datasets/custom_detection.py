dataset_type = 'DOTADataset'
data_root = 'C:/Users/CanAliYarman/Documents/MMrot/ETDII'  # Change this to your dataset path

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [  # Training pipeline
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='LoadAnnotations',  # Second pipeline to load annotations for current image
         with_bbox=True),  # Whether to use bounding box, True for detection
    dict(type='RResize',  # Augmentation pipeline that resize the images and their annotations
         img_scale=(1024, 1024)),  # The largest scale of image
    dict(type='RRandomFlip',  # Augmentation pipeline that flip the images and their annotations
         flip_ratio=0.5,  # The ratio or probability to flip
         version='oc'),  # The angle version
    dict(
        type='Normalize',  # Augmentation pipeline that normalize the input images
        mean=[123.675, 116.28, 103.53],  # These keys are the same of img_norm_cfg since the
        std=[58.395, 57.12, 57.375],  # keys of img_norm_cfg are used here as arguments
        to_rgb=True),
    dict(type='Pad',  # Padding config
         size_divisor=32),  # The number the padded images should be divisible
    dict(type='DefaultFormatBundle'),  # Default format bundle to gather data in the pipeline
    dict(type='Collect',  # Pipeline that decides which keys in the data should be passed to the detector
         keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(
        type='MultiScaleFlipAug',  # An encapsulation that encapsulates the testing augmentations
        img_scale=(1024, 1024),  # Decides the largest scale for testing, used for the Resize pipeline
        flip=False,  # Whether to flip images during testing
        transforms=[
            dict(type='RResize'),  # Use resize augmentation
            dict(
                type='Normalize',  # Normalization config, the values are from img_norm_cfg
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad',  # Padding config to pad images divisible by 32.
                 size_divisor=32),
            dict(type='DefaultFormatBundle'),  # Default format bundle to gather data in the pipeline
            dict(type='Collect',  # Collect pipeline that collect necessary keys for testing.
                 keys=['img'])
        ])
]

classes = ('DT')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='C:/Users/CanAliYarman/Documents/MMrot/ETDII/train/labels',
        img_prefix='C:/Users/CanAliYarman/Documents/MMrot/ETDII/train/images'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='C:/Users/CanAliYarman/Documents/MMrot/ETDII/val/labels',
        img_prefix='C:/Users/CanAliYarman/Documents/MMrot/ETDII/val/train/images'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='C:/Users/CanAliYarman/Documents/MMrot/ETDII/test/labels',
        img_prefix='C:/Users/CanAliYarman/Documents/MMrot/ETDII/test/images'))
