# coding=utf-8
# @Author:SDY
# @File:test9.py
# @Time:2023/2/26 16:35
# @Introduction:
a = input()
b = input()
if len(a) > len(b):
    a,b = b,a
res = ''
for i in range(len(a)):
    for j in range(i+1,len(a)+1):
        if (a[i:j] in b) and j-i >= len(res):
            res = a[i:j]
print(len(res))