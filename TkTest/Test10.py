# coding=utf-8
# @Author:SDY
# @File:Test10.py
# @Time:2024/7/29 19:39
# @Introduction:
import tkinter as tk
from tkinter import filedialog
import pandas as pd

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, df.head())

def analyze_trend():
    data = text_area.get(1.0, tk.END)
    # 这里假设数据已经以某种格式在text_area中
    # 你可以根据实际情况解析数据并调用你的分析函数
    # 例如，如果数据是CSV格式，你可以再次使用pandas读取
    df = pd.read_csv(pd.compat.StringIO(data))
    # 假设我们计算数据中的平均值作为趋势分析
    trend = df.mean()
    result_label.config(text=str(trend))

# 创建主窗口
root = tk.Tk()
root.title("数据趋势分析")

# 创建文件导入按钮
open_button = tk.Button(root, text="导入文件", command=open_file)
open_button.pack()

# 创建显示文件内容的文本区域
text_area = tk.Text(root)
text_area.pack()

# 创建分析趋势的按钮
analyze_button = tk.Button(root, text="分析趋势", command=analyze_trend)
analyze_button.pack()

# 创建显示分析结果的标签
result_label = tk.Label(root, text="分析结果将显示在这里")
result_label.pack()

# 运行主循环
root.mainloop()