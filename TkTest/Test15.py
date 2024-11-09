# coding=utf-8
# @Author:SDY
# @File:Test15.py
# @Time:2024/7/29 22:23
# @Introduction:
from tkinter import *
import tkinter.filedialog

def xz():
    fileName = tkinter.filedialog.askopenfilename()
    if fileName != '':
        lb.config(text='您选择的文件是'+fileName)
    else:
        lb.config(text='您没有选择任何文件')

root = Tk()
root.title('选择文件按钮')
root.geometry('500x300')
lb = Label(root,text='')
lb.pack()
btn = Button(root,text='弹出文件选择对话框',command=xz)
btn.pack()
root.mainloop()