import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def main():
    root = tk.Tk()
    root.title("Data-Driven UI")

    # 创建左右分隔的界面
    main_pane = ttk.PanedWindow(root, orient='horizontal')
    main_pane.grid(row=0, column=0, sticky='nsew')

    # 左侧框架
    left_frame = ttk.Frame(main_pane)
    left_frame.grid(row=0, column=0, sticky='nsew')
    main_pane.add(left_frame)

    # 右侧框架，用于显示图表
    right_frame = ttk.Frame(main_pane)
    right_frame.grid(row=0, column=1, sticky='nsew')
    main_pane.add(right_frame)

    # 左侧上下分隔的界面
    left_pane = ttk.PanedWindow(left_frame, orient='vertical')
    left_pane.grid(row=0, column=0, sticky='nsew')

    # 左上部分，显示设置参数
    top_frame = ttk.LabelFrame(left_pane, text="Settings")
    top_frame.grid(row=0, column=0, sticky='nsew')
    left_pane.add(top_frame)

    # 创建一个Table来显示数据
    settings_df = pd.DataFrame({
        'Temperature': [20, 22, 21],
        'Humidity': [45, 50, 47]
    })
    settings_tree = ttk.Treeview(top_frame, columns=list(settings_df.columns), show='headings')
    for col in settings_df.columns:
        settings_tree.heading(col, text=col)
        settings_tree.column(col, width=50)
    for index, row in settings_df.iterrows():
        settings_tree.insert('', 'end', values=list(row))
    settings_tree.grid(row=0, column=0, sticky='nsew')

    # 左下部分，显示日志数据
    bottom_frame = ttk.LabelFrame(left_pane, text="Logs")
    bottom_frame.grid(row=1, column=0, sticky='nsew')
    left_pane.add(bottom_frame)

    log_text = scrolledtext.ScrolledText(bottom_frame, width=50, height=10)
    log_text.insert('insert', "2023-09-17 10:00:00 - INFO: Started\n")
    log_text.insert('insert', "2023-09-17 10:05:00 - ERROR: Connection failed\n")
    log_text.grid(row=0, column=0, sticky='nsew')

    # 右侧图表展示
    # 创建Figure和多个子图
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))  # 2行1列的子图

    # 第一个图表
    axs[0].plot(settings_df['Temperature'], label='Temperature')
    axs[0].set_title('Temperature Trend')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].legend()

    # 第二个图表
    axs[1].plot(settings_df['Humidity'], label='Humidity')
    axs[1].set_title('Humidity Trend')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Humidity (%)')
    axs[1].legend()

    # 创建FigureCanvasTkAgg对象
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    # 用grid布局放置canvas
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    # 配置root和所有框架的行和列权重，以便它们可以扩展
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)
    top_frame.grid_rowconfigure(0, weight=1)
    top_frame.grid_columnconfigure(0, weight=1)
    bottom_frame.grid_rowconfigure(0, weight=1)
    bottom_frame.grid_columnconfigure(0, weight=1)
    right_frame.grid_rowconfigure(0, weight=1)
    right_frame.grid_columnconfigure(0, weight=1)

    root.mainloop()

if __name__ == '__main__':
    main()