# Projected LiDAR-DINO Contrastive Training

This directory contains one self-contained training script for the contrastive-learning experiment:

    RGB image -> pretrained MapAnything DINO feature map -> 1x1 projector
                                                                  \
                                                                   pixel InfoNCE
                                                                  /
    projected LiDAR (9 channels) -> project-defined dilated ResNet -> 1x1 projector

The script loads the Seq1LidarDataset and LiDAR projection code from
scripts/train_LiDAR+LoRA.py, including timestamp
matching, RGB quality filtering, calibrated LiDAR projection and the cache
format. Dataset defaults are retained: seq_len=4, stride=3, img_size=448,
tolerance=0.05, max_rgb_brightness=1 and max_rgb_contrast=1.

If a sequence has fewer valid views than the requested temporal window, the
contrastive adapter shortens that sequence's effective window to the number
of available views. It does not duplicate views. This is needed for current
sequences containing two valid RGB/LiDAR matches.

The script uses the same MapAnything RGB encoder and LiDAR encoder construction
as scripts/train_LiDAR+LoRA.py. This means the LiDAR branch uses the project's
own stride and dilation configuration, including the dilated-convolution change
that preserves the pixel-level feature-map resolution. It does not construct a
separate torchvision ResNet.

The projected input has the fixed layout used by the reference dataset:

    curvature, anisotropy, planarity,
    relative depth,
    normal x, normal y, normal z,
    valid pixel mask,
    depth edge

The RGB DINO encoder is frozen by default and loaded from the MapAnything
checkpoint in --model_dir. Use --train_dino to fine-tune it with a lower
learning rate. Pixel matches are sampled with --num_matches, whose default is
4096, following the OLIVINE pretraining configuration.

The loader defaults to --num_workers 0 because the reference projection uses
Open3D and the previous implementation observed worker segmentation faults in
this environment. Increasing it is supported but should be tested separately.

Example:

    CUDA_VISIBLE_DEVICES=3 python Contrastive/train_contrastive.py \
      --seq_roots /add02/users/xuyh/seq1/ /add02/users/xuyh/seq3/ \
      --model_dir /home/xuyh/mapanything/ \
      --cache_dir /add02/users/xuyh/cache/lidar_9ch_calib \
      --output_dir /add02/users/xuyh/checkpoints/contrastive_projected_lidar_resnet
