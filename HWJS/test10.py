# coding=utf-8
# @Author:SDY
# @File:test10.py
# @Time:2023/2/26 16:51
# @Introduction:
import math

def list_sum(lis,data):
    res = 0
    for d in lis:
        res += d
    return res != data

if __name__ == '__main__':
    m = int(input())
    data = m * m * m
    res = []
    tmp = -1
    while list_sum(res,data):
        tmp += 2
        init = tmp
        res.clear()
        for i in range(0,m):
            res.append(init)
            init += 2
    res_str = ''
    for i in range(0,len(res)):
        if i == 0:
            res_str = str(res[i])
        else:
            res_str = '{}+{}'.format(res_str,res[i])
    print(res_str)









