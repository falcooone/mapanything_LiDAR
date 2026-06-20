#!/usr/bin/env python3
#coding=gbk
"""
批量运行 eval.py，枚举不同的 seq_root 和 output_dir
"""

import os
import sys
import subprocess
import time

# ==========================================
# 1. 枚举参数（按索引一一对应）
# ==========================================
SEQ_ROOTS = [
     #"/add02/users/xuyh/seq1/shuangchuang_seq1_night1th",
     #"/add02/users/xuyh/seq1/shuangchuang_seq1_night2th",
     #"/add02/users/xuyh/seq1/shuangchuang_seq1_night3th",
     #"/add02/users/xuyh/seq1/shuangchuang_seq1_night4th",
    "/add02/users/xuyh/seq2/shuangchuang_seq2_night1th",
    "/add02/users/xuyh/seq2/shuangchuang_seq2_night2th",
    "/add02/users/xuyh/seq2/shuangchuang_seq2_night3th",
    "/add02/users/xuyh/seq2/shuangchuang_seq2_night4th",
    # "/add02/users/xuyh/seq3/shuangchuang_seq3_daytime1",
    # "/add02/users/xuyh/seq3/shuangchuang_seq3_daytime2",
    # "/add02/users/xuyh/seq3/shuangchuang_seq3_daytime3",
    # "/add02/users/xuyh/seq3/shuangchuang_seq3_daytime4",
    "/add02/users/xuyh/seq4/shuangchuang_seq4_daytime1th",
    "/add02/users/xuyh/seq4/shuangchuang_seq4_daytime2th",
    "/add02/users/xuyh/seq4/shuangchuang_seq4_daytime3th",
    "/add02/users/xuyh/seq4/shuangchuang_seq4_daytime4th",
]

OUTPUT_DIRS = [
    "/add02/users/xuyh/mapanything/output",
    "/add02/users/xuyh/mapanything/output",
    "/add02/users/xuyh/mapanything/output",
    "/add02/users/xuyh/mapanything/output",
    "/add02/users/xuyh/mapanything/output",
    "/add02/users/xuyh/mapanything/output",
    "/add02/users/xuyh/mapanything/output",
    "/add02/users/xuyh/mapanything/output",
]

# ==========================================
# 2. 固定参数（所有数据集共用）
# ==========================================
COMMON_ARGS = {
    "--model_dir": "/home/xuyh/mapanything/",
    #"--trained_ckpt": "/add02/users/xuyh/checkpoints/lora_rpe_weight/checkpoints/epoch_017.pt",
    "--trained_ckpt": "/add02/users/xuyh/checkpoints/lidar/checkpoints/best_full.pt",
    "--use_lidar": "1",
    "--use_lora": "0",
    "--lora_r": "16",
    "--lora_alpha": "16.0",
    "--batch_size": "4",
    "--img_size": "448",
    "--gpu": "3",
}

EVAL_PY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.py")

# ==========================================
# 3. 校验长度一致
# ==========================================
if len(SEQ_ROOTS) != len(OUTPUT_DIRS):
    print("Error: SEQ_ROOTS 与 OUTPUT_DIRS 长度不一致！")
    sys.exit(1)

# ==========================================
# 4. 遍历执行
# ==========================================
total = len(SEQ_ROOTS)
for i, (seq_root, output_dir) in enumerate(zip(SEQ_ROOTS, OUTPUT_DIRS), 1):
    part_name = os.path.basename(seq_root)
    print(f"\n{'='*60}")
    print(f"[{i}/{total}] 开始评测: {part_name}")
    print(f"  seq_root : {seq_root}")
    print(f"  output   : {output_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 构造命令
    cmd = [sys.executable, EVAL_PY_PATH]
    cmd.extend(["--seq_root", seq_root])
    cmd.extend(["--output_dir", output_dir])
    for k, v in COMMON_ARGS.items():
        cmd.extend([k, str(v)])

    # 执行
    start_t = time.time()
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{i}/{total}] {part_name} 运行失败，返回码: {e.returncode}")
        # 如需遇到错误继续下一个，注释掉下面的 sys.exit(1)
        sys.exit(1)

    elapsed = time.time() - start_t
    print(f"[{i}/{total}] {part_name} 评测完成，耗时 {elapsed:.1f}s")
    print(f"  结果保存至: {os.path.join(output_dir, 'evaluation_results.json')}")

    # 显存清理
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

print(f"\n{'='*60}")
print(f"全部 {total} 个数据集评测完成！")
print(f"{'='*60}")


