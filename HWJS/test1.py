# coding=utf-8
# @Author:SDY
# @File:test1.py
# @Time:2022/12/4 11:27
# @Introduction:

data = int(input())
tmp = 2
result = []
while data != 1:
    while data % tmp == 0:
        data = data / tmp
        result.append(str(tmp))
        result.append(" ")
    tmp = tmp + 1
for d in result:
    print(d,end="")

