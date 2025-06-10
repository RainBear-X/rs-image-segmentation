#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - 监督分类评估模块

功能：计算混淆矩阵、总体精度和 Kappa 系数，并生成混淆矩阵可视化图

输入：  
  - 预测结果数组（NumPy .npy 格式）  
  - 真实标签数组（NumPy .npy 格式）

输出：  
  - 混淆矩阵图像（PNG 格式）  
  - 控制台打印分类报告与统计结果  
  - 返回包含评估指标的字典

作者：Meng Yinan  
日期：2025-05-26
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import (
    confusion_matrix, classification_report,
    cohen_kappa_score, accuracy_score
)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.set_chinese_font import set_chinese_font

def evaluate_classification(prediction, ground_truth, class_names, save_dir=None):
    if save_dir is None:
        save_dir = ROOT_DIR / "output" / "supervised" / "evaluation"
    os.makedirs(save_dir, exist_ok=True)

    # Flatten for sklearn
    y_pred = prediction.flatten()
    y_true = ground_truth.flatten()

    # 过滤掉未标注区域（0）
    valid_mask = y_true > 0
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    # 准备 labels 列表，保证与 class_names 一一对应
    labels = list(range(1, len(class_names) + 1))

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # 打印报告，显式传入 labels
    print("🔎 分类报告：")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=3
    ))
    print(f"✅ 总体精度（OA）: {oa:.3f}")
    print(f"✅ Kappa 系数: {kappa:.3f}")

    # 绘图
    set_chinese_font()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("分类结果混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    print("📊 混淆矩阵已保存")

    return {
        "confusion_matrix": cm,
        "overall_accuracy": oa,
        "kappa": kappa
    }


if __name__ == '__main__':
    # —— 配置 —— #
    # 预测结果（H×W 整数数组，1,2,3…代表各类）
    pred_path = ROOT_DIR / 'output' / 'feature_outputs' / 'all_hierarchical_features.npy'
    # 真值（H×W 整数数组，0代表未标注，其余 1,2,3…对应类别）
    gt_path = ROOT_DIR / 'output' / 'ROI' / 'roi_mask.npy'
    # 类别名称列表，顺序要和 1,2,3… 对应
    class_names = ["水体", "植被", "建设用地"]

    # —— 加载 —— #
    prediction   = np.load(pred_path)
    ground_truth = np.load(gt_path)

    # —— 检查尺寸匹配 —— #
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"尺寸不匹配: prediction{prediction.shape} vs ground_truth{ground_truth.shape}"
        )

    # —— 评估 —— #
    results = evaluate_classification(
        prediction,
        ground_truth,
        class_names,
        save_dir=ROOT_DIR / 'output' / 'supervised' / 'evaluation'
    )

    # —— 打印 summary —— #
    print("\n=== 评估结果摘要 ===")
    print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
    print(f"Kappa Coefficient: {results['kappa']:.3f}")
    print("====================")
