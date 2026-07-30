#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总当前目录下所有 xlsx 评测结果，并按 LoRA / LiDAR 分组统计均值。
命名规则: {lora}_{lidar}_{seq}_{part}.xlsx
例如: 1_0_1_1.xlsx  -> 使用 LoRA, 不使用 LiDAR, seq1, part1
"""

import os
import re
import pandas as pd
from pathlib import Path
import numpy as np


# ==================== 配置 ====================
# 需要提取的指标列表
TARGET_METRICS = [
    "RRA@0.5°", "RRA@1.0°", "RRA@1.5°",
    "RTA@0.1m", "RTA@0.2m", "RTA@0.3m", "RTA@0.5m",
    "RPE_Trans_Mean", "RPE_Trans_Std",
    "RPE_Rot_Mean", "RPE_Rot_Std",
    # 如需更多指标，取消下面注释:
    # "ATE_RMSE",
    # "Abs_Rot_Mean", "Abs_Trans_Mean",
    # "Depth_Rel_Mean", "Depth_Tau_Mean",
    # "Ray_Error_Mean",
]

OUTPUT_FILE = "summary.xlsx"


# ==================== 文件名解析 ====================
def parse_filename(filename: str):
    """
    解析文件名: {lora}_{lidar}_{seq}_{part}.xlsx
    返回: dict 或 None
    """
    stem = Path(filename).stem
    parts = re.split(r'[_-]', stem)
    
    if len(parts) < 4:
        return None
    
    try:
        lora = int(parts[0])
        lidar = int(parts[1])
        seq = parts[2]
        part = parts[3]
        return {
            "use_lora": bool(lora),
            "use_lidar": bool(lidar),
            "seq": seq,
            "part": part,
            "filename": filename
        }
    except (ValueError, IndexError):
        return None


# ==================== 指标提取 ====================
def extract_metrics_from_excel(filepath: str):
    """
    从单个 xlsx 中提取目标指标。
    支持多列格式，自动识别第一列的指标名。
    """
    try:
        df = pd.read_excel(filepath, header=None)
    except Exception as e:
        print(f"[!] 读取失败: {filepath} -> {e}")
        return {}
    
    metrics = {}
    for _, row in df.iterrows():
        if row.empty or pd.isna(row.iloc[0]):
            continue
        
        cell_text = str(row.iloc[0]).strip()
        
        for metric in TARGET_METRICS:
            if metric in cell_text:
                # 在后续列中找第一个可转为 float 的值
                for val in row.iloc[1:]:
                    if pd.notna(val):
                        try:
                            metrics[metric] = float(val)
                            break
                        except (ValueError, TypeError):
                            continue
                break
    return metrics


# ==================== 打印辅助函数 ====================
def print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


# ==================== 主程序 ====================
def main():
    current_dir = Path(".")
    xlsx_files = sorted(current_dir.glob("*.xlsx"))
    
    if not xlsx_files:
        print("当前目录未找到 .xlsx 文件")
        return
    
    records = []
    
    print_section(f"扫描到 {len(xlsx_files)} 个 xlsx 文件")
    
    for fp in xlsx_files:
        # 跳过汇总文件自身
        if fp.name == OUTPUT_FILE:
            continue
            
        info = parse_filename(fp.name)
        if info is None:
            print(f"[!] 跳过 (命名不符合规则): {fp.name}")
            continue
        
        metrics = extract_metrics_from_excel(str(fp))
        
        record = {
            "文件名": fp.name,
            "LoRA": "是" if info["use_lora"] else "否",
            "LiDAR": "是" if info["use_lidar"] else "否",
            "Seq": info["seq"],
            "Part": info["part"],
            **metrics
        }
        records.append(record)
        
        flag = f"[LoRA={'1' if info['use_lora'] else '0'}|LiDAR={'1' if info['use_lidar'] else '0'}]"
        print(f"{flag} Seq{info['seq']}_Part{info['part']} -> 提取 {len(metrics)} 项指标")
    
    if not records:
        print("未提取到有效记录")
        return
    
    # 构建 DataFrame
    df = pd.DataFrame(records)
    
    # 数值列
    numeric_cols = [c for c in df.columns if c not in ["文件名", "LoRA", "LiDAR", "Seq", "Part"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 调整列顺序
    col_order = ["文件名", "LoRA", "LiDAR", "Seq", "Part"] + numeric_cols
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]
    
    # ==================== 分组统计 ====================
    print_section("分组统计结果")
    
    stats = {}
    
    # 1. 按 (LoRA, LiDAR) 组合分组 —— 均值 + 标准差 + 样本数
    print("\n【组合分组: LoRA × LiDAR (均值 | 标准差 | 样本数)】")
    group_combo = df.groupby(["LoRA", "LiDAR"])[numeric_cols].agg(['mean', 'std', 'count'])
    group_combo.columns = [f"{col}_{stat}" for col, stat in group_combo.columns]
    print(group_combo.round(4).to_string())
    stats["组合分组_详细"] = group_combo.reset_index()
    
    # 组合分组 —— 仅均值（更简洁）
    print("\n【组合分组: LoRA × LiDAR (仅均值)】")
    group_combo_mean = df.groupby(["LoRA", "LiDAR"])[numeric_cols].mean().round(4)
    print(group_combo_mean.to_string())
    stats["组合分组_均值"] = group_combo_mean.reset_index()
    
    # 2. 单独按 LoRA 分组
    print("\n【按 LoRA 分组 (均值)】")
    group_lora = df.groupby(["LoRA"])[numeric_cols].mean().round(4)
    print(group_lora.to_string())
    stats["LoRA分组"] = group_lora.reset_index()
    
    # 3. 单独按 LiDAR 分组
    print("\n【按 LiDAR 分组 (均值)】")
    group_lidar = df.groupby(["LiDAR"])[numeric_cols].mean().round(4)
    print(group_lidar.to_string())
    stats["LiDAR分组"] = group_lidar.reset_index()
    
    # 4. 按 Seq 分组（看不同序列的表现）
    print("\n【按 Seq 分组 (均值)】")
    group_seq = df.groupby(["Seq"])[numeric_cols].mean().round(4)
    print(group_seq.to_string())
    stats["Seq分组"] = group_seq.reset_index()
    
    # ==================== 保存到 Excel (多 Sheet) ====================
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # Sheet 1: 详细结果
        df.to_excel(writer, sheet_name="详细结果", index=False)
        
        # Sheet 2: 组合分组均值
        stats["组合分组_均值"].to_excel(writer, sheet_name="组合分组均值", index=False)
        
        # Sheet 3: 组合分组均值+标准差+样本数
        stats["组合分组_详细"].to_excel(writer, sheet_name="组合分组统计", index=False)
        
        # Sheet 4: LoRA 单独分组
        stats["LoRA分组"].to_excel(writer, sheet_name="LoRA分组", index=False)
        
        # Sheet 5: LiDAR 单独分组
        stats["LiDAR分组"].to_excel(writer, sheet_name="LiDAR分组", index=False)
        
        # Sheet 6: Seq 分组
        stats["Seq分组"].to_excel(writer, sheet_name="Seq分组", index=False)
    
    print_section(f"汇总完成！已保存至: {OUTPUT_FILE}")
    print("\nExcel 文件包含以下 Sheet:")
    print("  1. 详细结果      - 每个文件提取的原始指标")
    print("  2. 组合分组均值   - 按 LoRA×LiDAR 分组求均值 (4 种组合)")
    print("  3. 组合分组统计   - 按 LoRA×LiDAR 分组求均值+标准差+样本数")
    print("  4. LoRA分组     - 仅按是否使用 LoRA 分组求均值")
    print("  5. LiDAR分组    - 仅按是否使用 LiDAR 分组求均值")
    print("  6. Seq分组      - 按 Seq 分组求均值")


if __name__ == "__main__":
    main()