# coding=utf-8
# @Author:SDY
# @File:test21.py
# @Time:2023/3/12 12:09
# @Introduction:
if __name__ == '__main__':
    s = input()
    d = s.split(".")
    if int(d[1][0]) >= 5:
        print(int(d[0])+1)
    else:
        print(int(d[0]))
