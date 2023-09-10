# coding=utf-8
# @Author:SDY
# @File:test15.py
# @Time:2023/3/12 10:51
# @Introduction:
import re
if __name__ == '__main__':
    while True:
        s = input()
        data = re.findall(r"[A-Z]",s)
        print(len(data))

