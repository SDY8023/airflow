import os
import sys
import tkinter as tk

from PIL import Image,ImageTk
from tkinter import filedialog, scrolledtext,ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from RoadPrediction import LASAMP
from RoadPrediction.LASAMP import RoadPredictionParam


class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("400x300")  # 设置窗口大小
        self.filename = None
        self.parameters = {}
        self.default_values = {'num_epochs': '200',
                               'test_row':'5',
                               'prediction_years':'1',
                               'hidden_size':'50',
                               'reduction':'sum',
                               'lr':'0.01',
                               'num_layers':'1',
                               'aa':'10'}
        self.param_descriptions = {
            "num_epochs": "模型训练次数",
            "test_row": "以最后多少年数据为测试集",
            "prediction_years": "预测年数",
            "hidden_size": "隐藏层大小",
            "reduction": "损失函数的归约方式",
            "lr": "学习率",
            "num_layers":"LSTM层数",
            "aa":"aa"
        }
        self.create_init_page()
        self.function_window = None
        self.param_window = None

    def create_init_page(self):
        # 创建初始化页面
        self.select_button = tk.Button(self.root,bg='#ADD8E6', text="道路预测分析", command=self.create_param_page, font=("Arial", 16), height=1, width=10)
        self.select_button.place(relx=0.5, rely=0.3, anchor=tk.CENTER)


    def select_fuction(self):
        self.function_window = self.root
        self.clear_window(self.function_window)
        menubar = tk.Menu(self.function_window)
        self.function_window.config(menu=menubar)
        function_menubar = tk.Menu(menubar)

    def create_param_page(self):
        # 设置数据参数
        self.function_window = self.root
        self.clear_window(self.function_window)
        self.function_window.title("功能选择")
        self.function_window.geometry("600x400")
        self.select_button = tk.Button(self.function_window, text="选择文件", command=self.select_file, font=("Arial", 16), height=1, width=10)
        self.select_button.grid(row=0, column=0, sticky="ew")
        # 显示选择的文件名的Label
        self.filename_label = tk.Label(self.function_window, text="", font=("Arial", 16))
        self.filename_label.grid(row=0, column=1, sticky="ew")


    def select_file(self):
        self.filename = filedialog.askopenfilename()
        self.filename_label.config(text=f"已选择文件: {self.filename}")
        if self.filename:
            self.param_window = tk.Toplevel(self.root)
            self.param_window.title("分析参数设置")
            self.create_parameter_entries()
            self.show_parameters()
            self.analyze_button = tk.Button(self.param_window,text="开始分析",command=self.analyze,font=("Arial", 16), height=2, width=15)
            self.analyze_button.grid(row=50, column=0, sticky="ew")

    def create_parameter_entries(self):
        for idx, param in enumerate(self.default_values, start=2):  # 起始行号为2
            label = tk.Label(self.param_window, text=f"{param}: ", font=("Arial", 16))
            label.grid(row=idx, column=0, sticky="w")

            entry = tk.Entry(self.param_window, font=("Arial", 16))
            entry.insert(0, self.default_values[param])  # 设置默认值
            entry.config(disabledforeground="grey")  # 设置灰色字体
            entry.grid(row=idx, column=1, sticky="ew")
            self.parameters[param] = entry

            # 创建参数说明标签
            description = tk.Label(self.param_window, text=self.param_descriptions[param], font=("Arial", 12),justify="left")
            description.grid(row=idx, column=2, sticky="w")

    def show_parameters(self):
        for param, entry in self.parameters.items():
            label = tk.Label(self.param_window, text=f"{param}: ", font=("Arial", 16))
            label.grid(row=list(self.parameters.keys()).index(param) + 2, column=0, sticky="w")
            entry.config(state=tk.NORMAL)  # 允许编辑
            entry.grid(row=list(self.parameters.keys()).index(param) + 2, column=1, sticky="ew")

    def hide_parameters(self,window):
        for param, entry in self.parameters.items():
            label = tk.Label(window, text=f"{param}: ", font=("Arial", 16))
            label.grid_remove()

            entry.config(state=tk.DISABLED)  # 禁止编辑
            entry.grid_remove()

    def analyze(self):
        if self.filename:
            # 销毁窗口前将参数保留下来
            params = {param: entry.get() for param, entry in self.parameters.items()}
            self.param_window.destroy()
            # 创建一个新的窗口来显示分析日志
            show_window = tk.Toplevel(self.function_window)
            show_window.title("结果展示")
            show_main_pane = tk.PanedWindow(show_window,orient='horizontal')
            show_main_pane.grid(row=0,column=0,sticky='nsew')
            # 左侧框架 显示参数和日志
            left_frame = tk.Frame(show_main_pane)
            show_main_pane.add(left_frame)
            # 右侧框架 显示图片
            right_frame = tk.Frame(show_main_pane)
            show_main_pane.add(right_frame)
            # 左侧上下分隔的界面
            left_pane = tk.PanedWindow(left_frame, orient='vertical')
            left_pane.grid(row=0,column=0,sticky='nsew')
            # 左上部分，显示设置参数
            top_frame = tk.LabelFrame(left_pane,text='params')
            top_frame.grid(row=0, column=0, sticky='nsew')
            left_pane.add(top_frame)
            param_df = pd.DataFrame(list(params.items()),columns=['param','value'])
            param_df['description'] = param_df['param'].map(self.param_descriptions)
            param_tree = ttk.Treeview(top_frame,columns=list(param_df.columns),show='headings')
            for col in param_df.columns:
                param_tree.heading(col, text=col)
                param_tree.column(col, width=50)
            for index, row in param_df.iterrows():
                param_tree.insert('', 'end', values=list(row))
            # 使用grid布局，且确保treeview能够填充整个框架
            param_tree.grid(row=0, column=0, sticky='nsew')

            # 左下部分，显示日志数据
            bottom_frame = tk.LabelFrame(left_pane,text='logs')
            left_pane.add(bottom_frame)  # 确保框架能够扩展
            log_text = scrolledtext.ScrolledText(bottom_frame, width=50, height=10)

            # 解析文件
            log_text.insert(tk.END,f"开始解析文件:{self.filename}\n")
            data_dict = self.parse_file_data()
            log_text.insert(tk.END,f"文件解析成功!\n")
            # 在这里添加文件分析的代码，并将日志输出到Text控件中
            log_text.insert(tk.END, f"开始分析数据\n")
            analysis_param = RoadPredictionParam()
            for key, value in params.items():
                setattr(analysis_param, key, value)
            road_pre = LASAMP.RoadPrediction()
            # 预测结果展示
            predict_result = road_pre.predictedData(data_dict,log_text,analysis_param)
            loss_values = predict_result[0]
            target_list = predict_result[1]
            final_result_list = predict_result[2]
            # 确保ScrolledText能够填充整个框架
            log_text.grid(row=0,column=0,sticky='nsew')
            log_text.config(state=tk.DISABLED)  # 禁止编辑
            # 右侧展示预测结果图表
            fig, axs = plt.subplots(2, 1, figsize=(10, 5))  # 2行1列的子图
            axs[0].plot(loss_values)
            axs[0].set_title('训练损失曲线')
            axs[0].set_xlabel('训练轮次')
            axs[0].set_ylabel('损失值')
            axs[0].legend()

            axs[1].plot(final_result_list, label='预测PQI', marker='o')
            axs[1].plot(target_list, label='实际PQI', marker='x')
            axs[1].set_title('预测值与实际值对比')
            axs[1].set_xlabel('样本索引')
            axs[1].set_ylabel('PQI值')
            axs[1].legend()
            # 创建FigureCanvasTkAgg对象
            canvas = FigureCanvasTkAgg(fig, master=right_frame)
            canvas.draw()
            # 用grid布局放置canvas
            canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')


            left_frame.grid_rowconfigure(0, weight=1)
            left_frame.grid_columnconfigure(0, weight=1)
            top_frame.grid_rowconfigure(0, weight=1)
            top_frame.grid_columnconfigure(0, weight=1)
            bottom_frame.grid_rowconfigure(0, weight=1)
            bottom_frame.grid_columnconfigure(0, weight=1)
            right_frame.grid_rowconfigure(0, weight=1)
            right_frame.grid_columnconfigure(0, weight=1)

            print("分析完成")
        else:
            print("请先选择文件")
        # 关闭参数窗口
        self.param_window.destroy()

    # 解析数据源
    def parse_file_data(self):
        df = pd.read_excel(self.filename)
        year = df['年份'].tolist()
        temperature = df['温度'].tolist()
        traffic_volume = df['交通量'].tolist()
        passenger_freight = df['客货比'].tolist()
        pqi = df['PQI'].tolist()
        data_dict = {'年份': year,
                     '温度': temperature,
                     '交通量': traffic_volume,
                     '客货比': passenger_freight,
                     'PQI': pqi}
        return data_dict

    def clear_window(self,window):
        for widget in window.winfo_children():
            widget.destroy()

def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

root = tk.Tk()
root.title('道路预测')
img = Image.open(resource_path("p1.jfif"))  # 请替换为你的图片路径
bg_img = ImageTk.PhotoImage(img)
bg_label = tk.Label(root, image=bg_img)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
app = AnalysisApp(root)
root.mainloop()