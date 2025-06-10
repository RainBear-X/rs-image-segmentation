#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - 中文字体设置模块

功能：为 Matplotlib 配置中文字体，确保绘图时中文正常显示

输入：无  
输出：全局 Matplotlib 字体已设置为指定中文字体

作者：Meng Yinan  
日期：2025-05-26
'''

from pathlib import Path
from .path_utils import get_project_root

def set_chinese_font():
    try:
        print("🎮设置中文字体...")
        font_path = get_project_root() / "fonts" / "STHeiti Medium.ttc"
        if font_path.is_file():
            from matplotlib import font_manager
            prop = font_manager.FontProperties(fname=str(font_path))
            import matplotlib.pyplot as plt
            plt.rcParams["font.family"] = prop.get_name()
        else:
            print(f"⚠️ 中文字体文件未找到：{font_path}")
    except Exception as e:
        print("⚠️ 设置中文字体失败：", e)
