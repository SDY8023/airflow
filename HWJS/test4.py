# coding=utf-8
# @Author:SDY
# @File:test4.py
# @Time:2023/2/26 11:27
# @Introduction:
import re
str1 = input()
str2 = input()

len1 = len(str1)
len2 = len(str2)
i = 0
j = 0
rs = True
if str1.find('?') == -1 and str1.find('*') == -1 and len1 != len2:
    rs = False
else:
    while(i < len1 and j < len2):
        d1 = str1[i]
        if d1 not in ['?','*']:
            d2 = str2[j]
            if d1 != d2:
                rs = False
                i = len1
            else:
                i += 1
                j += 1
        elif d1 == '?' and str2[j] != '.':
            i += 1
            j += 1
        elif d1 == '?' and str2[j] == '.':
            rs = False
            i = len1
        elif d1 == '*' and str2[j] != '.':
            if j < len2:
                j += 1
        elif d1 == '*' and str2[j] == '.':
            rs = False
            i = len1
        else:
            i += 1
            j += 1
print(str(rs).lower())







