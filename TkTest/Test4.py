# coding=utf-8
# @Author:SDY
# @File:Test4.py
# @Time:2024/7/22 22:38
# @Introduction:
from tkinter import *

def run1():
    a = float(inp1.get())
    b = float(inp2.get())
    s = '%0.2f+%0.2f=%0.2f\n' % (a,b,a+b)
    txt.insert(END,s)
    inp1.delete(0,END)
    inp2.delete(0,END)

def run2(x, y):
    a = float(x)
    b = float(y)
    s = '%0.2f+%0.2f=%0.2f\n' % (a, b, a + b)
    txt.insert(END,s)
    inp1.delete(0,END)
    inp2.delete(0,END)

root = Tk()
root.geometry('460x240')
root.title('简单加法')

lb1 = Label(root,text='请输入两个数,按下面任意按钮进行计算')
lb1.place(relx=0.1,rely=0.1,relwidth=0.8,relheight=0.1)
inp1 = Entry(root)
inp1.place(relx=0.1,rely=0.2,relwidth=0.3,relheight=0.1)
inp2 = Entry(root)
inp2.place(relx=0.6,rely=0.2,relwidth=0.3,relheight=0.1)

btn1 = Button(root,text='方法一',command=run1)
btn1.place(relx=0.1,rely=0.4,relwidth=0.3,relheight=0.1)

btn2 = Button(root,text='方法二',command=lambda: run2(inp1.get(),inp2.get()))
btn2.place(relx=0.6,rely=0.4,relwidth=0.3,relheight=0.1)

txt = Text(root)
txt.place(rely=0.6,relheight=0.4)

root.mainloop()