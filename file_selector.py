import tkinter as tk
from tkinter import filedialog
import os


def select_image():
    """
    弹出文件选择对话框，返回选中的图片路径。
    如果用户取消选择，则返回 None。
    """
    # 创建一个隐藏的 Tkinter 根窗口
    root = tk.Tk()
    root.withdraw()

    # 强制窗口置顶，防止被 IDE 或其他窗口遮挡
    root.attributes('-topmost', True)

    # 弹出选择对话框
    file_path = filedialog.askopenfilename(
        title="请选择要识别的图片",
        filetypes=[
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("所有文件", "*.*")
        ]
    )

    # 销毁临时窗口资源
    root.destroy()

    if file_path and os.path.exists(file_path):
        return file_path
    return None