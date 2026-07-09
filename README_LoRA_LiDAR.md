# MapAnything LoRA-LiDAR Extension

基于 [MapAnything](https://github.com/naver/mapanything) 的 LiDAR 模态扩展与 LoRA 参数高效微调实现。

## 与原始 MapAnything 的区别

本仓库在 MapAnything 基础上新增以下能力：

- **LiDAR 模态支持**：基于 FiLM 的绝对尺度条件化融合与零初始化融合卷积
- **LoRA 参数高效微调**：RGB 端 LoRA 微调 + LiDAR 端全参数训练的解耦范式
- **训练稳定性优化**：LoRA+ 非对称学习率（B/A 16:1）、EMA 稳定机制、分层调度
- **损失函数全面修复**：对数空间深度损失、Chordal 旋转距离、鲁棒回归、Top-N 像素排除

原始 MapAnything 的 README 请参见 [README.md](./README.md)。

## 环境安装

```bash
pip install torch torchvision safetensors scipy pillow numpy open3d
# 需预先安装 MapAnything 及其依赖
```

## 项目结构

```
.
├── scripts/
│   ├── LoRA.py                    # 纯RGB LoRA训练
│   ├── train_LiDAR+LoRA.py        # LiDAR融合训练
│   ├── test.py                    # 从数据集测试脚本
│   └── test_LoRA_LiDAR.py         # 批量训练脚本（调用test.py）
├── mapanything/
│   ├── models                     
        └── mapanything
            └── model.py           # 更新后有LiDAR编码融合的模型代码
    └── utils/
│       └── inference.py           # 更新ALLOWED_VIEW_KEYS
└── README.md                      # 原始MapAnything说明 (保留)
└── README_LoRA_LiDAR.md           # 本扩展说明 (此文件)
```

## 快速开始

### 纯RGB LoRA训练

```bash
python scripts/LoRA.py \
    --seq_root /path/to/seq1 /path/to/seq3 \
    --model_dir /path/to/mapanything \
    --output_dir ./checkpoints/lora_rgb \
    --lr 5e-5 --lr_b_multiplier 16.0 \
    --lora_r 8 --lora_alpha 8 \
    --use_ema --grad_clip 1.0
```

### LiDAR融合训练

```bash
python scripts/train_lidar_fusion.py \
    --seq_root /path/to/seq1 \
    --model_dir /path/to/mapanything \
    --output_dir ./checkpoints/lidar_fusion \
    --lr 5e-5 --lora_r 8 --lora_alpha 16
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **FiLM绝对尺度融合** | 利用LiDAR深度标量对RGB特征进行条件化调制，缓解单目尺度模糊 |
| **零初始化融合卷积** | RGB侧恒等初始化、LiDAR侧置零，实现渐进式多模态融合 |
| **LoRA差异化分组训练** | A/B矩阵非对称学习率（16:1）+ 编码器/头部分层调度 |
| **训练稳定性优化** | EMA指数移动平均、FP32安全路径、紧缩梯度裁剪 |
| **工程支持** | DDP分布式训练、多序列ConcatDataset、LoRA-only检查点热切换 |

## 致谢

本项目基于 MapAnything 开发，感谢原作者的开源工作。
