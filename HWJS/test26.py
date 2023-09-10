# coding=utf-8
# @Author:SDY
# @File:test26.py
# @Time:2023/3/15 21:43
# @Introduction:

if __name__ == '__main__':
    s = input()
    d = bin(int(s))[2:]
    max_result = 0
    datas = d.split("0")
    for m in datas:
        if max_result < len(m):
            max_result = len(m)
    print(max_result)