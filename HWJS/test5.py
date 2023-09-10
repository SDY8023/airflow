# coding=utf-8
# @Author:SDY
# @File:test5.py
# @Time:2023/2/26 14:49
# @Introduction:
import re
while 1:
    try:
        a = input().lower()
        b = input().lower()
        a = a.replace('.','\.').replace('?','[a-z0-9]{1}').replace('*','[a-z0-9]*')
        print(a)
        if b in re.findall(a,b):
            print('true')
        else:
            print('false')
    except:
        break