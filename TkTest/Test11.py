# coding=utf-8
# @Author:SDY
# @File:Test11.py
# @Time:2024/7/29 19:41
# @Introduction:
from tkinter import *

def new(lb1:Label):
    s = '新建'
    lb1.config(text=s)

def ope(lb1:Label):
    s = '打开'
    lb1.config(text=s)

def sav(lb1:Label):
    s = '保存'
    lb1.config(text=s)

def cut(lb1:Label):
    s = '剪切'
    lb1.config(text=s)

def pas(lb1:Label):
    s = '粘贴'
    lb1.config(text=s)

def cop(lb1:Label):
    s = '复制'
    lb1.config(text=s)

def popupmenu(event):
    mainmenu.post(event.x_root,event.y_root)


if __name__ == '__main__':
    root = Tk()
    root.title('菜单实验')
    root.geometry('320x240')

    lb1 = Label(root,text='显示信息',font=('黑体',32,'bold'))
    lb1.pack()

    mainmenu = Menu(root)
    menuFile = Menu(mainmenu)
    menuEdit = Menu(mainmenu)
    mainmenu.add_cascade(label='文件',menu=menuFile)
    menuFile.add_command(label='新建',command=new(lb1))
    menuFile.add_command(label='打开', command=ope(lb1))
    menuFile.add_command(label='保存', command=sav(lb1))
    menuFile.add_cascade(label='编辑', command=menuEdit)
    menuFile.add_command(label='剪切', command=cut(lb1))
    menuFile.add_command(label='复制', command=cop(lb1))
    menuFile.add_command(label='粘贴', command=pas(lb1))
    menuFile.add_separator() # 分割线
    menuFile.add_command(label='退出', command=root.destroy)

    root.config(menu=mainmenu)
    root.bind('<Button-3>',popupmenu)
    root.mainloop()