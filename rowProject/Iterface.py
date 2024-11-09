# coding=utf-8
# @Author:SDY
# @File:Iterface.py
# @Time:2024/8/3 17:13
# @Introduction:
from tkinter import *
import tkinter.filedialog
def xz(lb:Label):
    fileName = tkinter.filedialog.askopenfilename()
    if fileName != '' and is_excel_file(fileName):
        lb.config(text='您选择的文件是'+fileName)
    else:
        lb.config(text='您没有选择任何文件')

# 判断文件类型是否为excel
def is_excel_file(filename:str):

    return filename.lower().endswith(('.xls', '.xlsx', '.xlsm', '.xlsb', '.xla', '.xlam', '.xlt', '.xltx', '.xltm'))


if __name__ == '__main__':
    root = Tk()
    root.title("道路预测")
    root.geometry('500x300')
    # 设置导入数据文件按钮
    file_lb = Label(root, text='',wraplength=450)
    # 设置导入数据按钮
    btn = Button(root, text='请选择您的数据文件', command=lambda: xz(file_lb))
    btn.pack()
    file_lb.pack()
    root.mainloop()
