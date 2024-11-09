# coding=utf-8
# @Author:SDY
# @File:Test16.py
# @Time:2024/7/29 22:27
# @Introduction:
from tkinter import *
import tkinter.colorchooser

def xz():
    color = tkinter.colorchooser.askcolor()
    colorstr =  str(color)
    print('打印字符串%s 切掉=%s' % (colorstr,colorstr[-9:2]))
    lb.config(text=colorstr[-9:2],background=colorstr[-9:-2])

root = Tk()
root.title('颜色选择器')
root.geometry('300x300')
lb = Label(root,text='请关注颜色变化')
lb.pack()
btn = Button(root,text='弹出颜色选择对话框',command=xz)
btn.pack()
root.mainloop()