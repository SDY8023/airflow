# coding=utf-8
# @Author:SDY
# @File:Test7.py
# @Time:2024/7/28 9:01
# @Introduction:
from tkinter import *
def ini():
    Lstbox1.delete(0,END)
    list_items = ['数学','物理','化学','语文','外语']
    for item in list_items:
        Lstbox1.insert(END,item)

def clear():
    Lstbox1.delete(0,END)

def ins():
    if entry.get() != '':
        if Lstbox1.curselection() == ():
            Lstbox1.insert(Lstbox1.size(),entry.get())
        else:
            Lstbox1.insert(Lstbox1.curselection(),entry.get())

def updt():
    if entry.get() != '' and Lstbox1.curselection() != ():
        selected=Lstbox1.curselection()[0]
        Lstbox1.delete(selected)
        Lstbox1.insert(selected,entry.get())

def delt():
    if Lstbox1.curselection() != ():
        Lstbox1.delete(Lstbox1.curselection())



root = Tk()
root.title('列表实验')
root.geometry('320x240')

frame1 = Frame(root,relief=GROOVE)
frame1.place(relx=0.0)

frame2 = Frame(root,relief=GROOVE)
frame2.place(relx=0.5)

Lstbox1=Listbox(frame1)
Lstbox1.pack()

entry = Entry(frame2)
entry.pack()

btn1 = Button(frame2,text='初始化',command=ini)
btn1.pack(fill=X)

btn2 = Button(frame2,text='添加',command=ins)
btn2.pack(fill=X)

btn3 = Button(frame2,text='插入',command=ins)
btn3.pack(fill=X)

btn4 = Button(frame2,text='修改',command=updt)
btn4.pack(fill=X)

btn5 = Button(frame2,text='删除',command=delt)
btn5.pack(fill=X)

btn6 = Button(frame2,text='清空',command=clear)
btn6.pack(fill=X)

root.mainloop()