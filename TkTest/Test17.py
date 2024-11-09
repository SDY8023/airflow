# coding=utf-8
# @Author:SDY
# @File:Test17.py
# @Time:2024/8/19 21:54
# @Introduction:

import tkinter as tk
from PIL import Image, ImageTk
import os
print(os.getcwd())
# 创建主窗口
root = tk.Tk()
root.title("道路预测分析系统")

# 加载背景图片
img = Image.open("background\\UDDPPGY2.png")  # 请替换为你的图片路径
bg_img = ImageTk.PhotoImage(img)

# 创建一个Label并使用背景图片填充
bg_label = tk.Label(root, image=bg_img)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# 创建按钮
button = tk.Button(root, text="道路预测分析", font=("Arial", 16), relief="raised", bd=3)
# 使用place布局管理器，将按钮置于窗口中央
button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# 运行主窗口事件循环
root.mainloop()
