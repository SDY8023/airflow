# coding=utf-8
# @Author:SDY
# @File:test2.py
# @Time:2022/12/4 14:48
# @Introduction:
import sys

result = {}
for line in sys.stdin:
    for i in range(0,int(line)):
        data = input()
        a = data.split(" ")
        if int(a[0]) in result:
            result[int(a[0])] = int(a[1]) + result.get(int(a[0]))
        else:
            result[int(a[0])] = int(a[1])
    for k in sorted(result):
        print(str(k) + " " + str(result.get(k)))