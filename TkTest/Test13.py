# coding=utf-8
# @Author:SDY
# @File:Test13.py
# @Time:2024/7/29 22:14
# @Introduction:
from tkinter import *
import tkinter.messagebox

def xz(lb:Label):
    answer = tkinter.messagebox.askokcancel('请选择','请选择确定或取消')
    if answer:
        lb.config(text='已确认')
    else:
        lb.config(text='已取消')


if __name__ == '__main__':
    root = Tk()
    root.title('弹出对话框测试')
    root.geometry('500x500')
    lb = Label(root,text='')
    lb.pack()
    btn = Button(root,text='弹出对话框',command=xz(lb))
    btn.pack()
    root.mainloop()