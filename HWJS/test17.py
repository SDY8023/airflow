# coding=utf-8
# @Author:SDY
# @File:test17.py
# @Time:2023/3/12 11:07
# @Introduction:
import re
if __name__ == '__main__':
    s1 = input()
    s2 = input()
    result = re.findall(s2.lower(),s1.lower())
    print(len(result))