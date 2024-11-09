# coding=utf-8
# @Author:SDY
# @File:Test12.py
# @Time:2024/7/29 20:04
# @Introduction:
from tkinter import *

def newwind():
    winNew = Toplevel(root)
    winNew.geometry('320x240')
    winNew.title('新窗体')
    lb2 = Label(root,text='我在新窗体上')
    lb2.place(relx=0.2,rely=0.2)
    btClose=Button(winNew,text='关闭',command=winNew.destroy)
    btClose.place(relx=0.7,rely=0.5)

if __name__ == '__main__':
    root = Tk()
    root.title('新建窗体实验')
    root.geometry('500x500')
    lb1 = Label(root,text='主窗体',font=('黑体',32,'bold'))
    lb1.place(relx=0.2,rely=0.2)

    main_menu = Menu(root)
    menu_file = Menu(main_menu)
    main_menu.add_cascade(label='菜单',menu=menu_file)
    menu_file.add_command(label='新窗体',command=newwind)
    menu_file.add_separator()
    menu_file.add_command(label='退出',command=root.destroy)

    root.config(menu=main_menu)
    root.mainloop()