# Contrastive Training

This folder contains a standalone contrastive-training pipeline for aligning
the project's RGB DINO encoder with the LiDAR ResNet encoder.

The original project code is not modified.

## What it does

- Loads the existing `MapAnything` checkpoint/config from `--model_dir`
- Uses the standalone `Contrastive/` dataset implementation directly
- Trains only:
  - RGB encoder
  - LiDAR encoder
  - projection heads
- Uses symmetric InfoNCE in the style of OLIVINE

## Example

```bash
python Contrastive/train_contrastive.py ^
  --model_dir output/your_model_dir ^
  --seq_roots seq1 ^
  --output_dir Contrastive/output/run1 ^
  --batch_size 16 ^
  --epochs 20 ^
  --lr 1e-5 ^
  --proj_lr 1e-4 ^
  --max_samples 4 ^
  --lidar_lr_scale 0.5
```

## Notes

- `--max_samples` is applied per dataset root, because the underlying
  `Seq1LidarDataset` already implements that limit.
- For stronger LiDAR adaptation, try a larger `--lidar_lr_scale` than `1.0`.
- The default contrastive pairing is one RGB-LiDAR pair per sample with batch
  negatives.
- The standalone trainer now defaults to `--amp`, `--grad_checkpointing`, and
  `--empty_cache_each_step` to reduce peak GPU memory.
