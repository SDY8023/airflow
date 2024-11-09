# coding=utf-8
# @Author:SDY
# @File:Test1.py
# @Time:2024/7/22 22:06
# @Introduction:
from tkinter import *
import time
def testLable():
    root = Tk()
    root.title("预测模型")
    lb1 = Label(root,text='第一个标签',
               bg='#d3fbfb',
               fg='red',
               font=('宋体',32),
               width=10,
               heigh=2,
               relief=GROOVE)
    lb1.grid(column=2,row=0)
    lb2 = Label(root,text='第二个标签',
               bg='#d3fbfb',
               fg='green',
               font=('宋体',32),
               width=10,
               heigh=2,
               relief=GROOVE)
    lb2.grid(column=0,row=1)
    lb3 = Label(root,text='第三个标签',
               bg='#d3fbfb',
               fg='blue',
               font=('宋体',32),
               width=10,
               heigh=2,
               relief=GROOVE)
    lb3.grid(column=1,row=2)
    root.mainloop()

def testConfigure():
    root = Tk()
    root.title("时钟")
    lb = Label(root,text='',fg='blue',font=('黑体',80))
    lb.pack()
    getTime(lb,root)
    root.mainloop()

def getTime(lb,root):
    timestr = time.strftime("%H:%M:%S")
    lb.configure(text=timestr)
    root.after(1000,getTime(lb,root))
if __name__ == '__main__':
    testConfigure()
