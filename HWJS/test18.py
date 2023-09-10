# coding=utf-8
# @Author:SDY
# @File:test18.py
# @Time:2023/3/12 11:10
# @Introduction:
if __name__ == '__main__':
    s = input()
    d = divmod(len(s),8)
    consult = d[0]
    remainder = d[1]
    if remainder != 0:
        for i in range(8-remainder):
            s = s + "0"
    while len(s) != 0:
        print(s[:8])
        s = s[8:]