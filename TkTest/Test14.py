# coding=utf-8
# @Author:SDY
# @File:Test14.py
# @Time:2024/7/29 22:20
# @Introduction:
from tkinter import *
import tkinter.messagebox
from tkinter.simpledialog import askstring


def xz():
    s = askstring('请输入','请输入文字')
    lb.config(text=s)


root = Tk()
root.title('弹出对话框测试')
root.geometry('500x500')
btn = Button(root, text='弹出对话框', command=xz)
btn.pack()
lb = Label(root, text='')
lb.pack()
root.mainloop()
