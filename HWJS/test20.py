# coding=utf-8
# @Author:SDY
# @File:test20.py
# @Time:2023/3/12 11:48
# @Introduction:
import math
def check(n):
    number = n
    result = True
    for i in range(int(math.sqrt(n))+1):
        if i > 1 and divmod(number,i)[1] == 0:
            result = False
            break
    return result

if __name__ == '__main__':
    s = int(input())
    result = []
    i = 2
    while not check(s):
        data = divmod(s, i)
        if check(i) and data[1] == 0:
            s = data[0]
            result.append(i)
        else:
            i += 1
    result.append(s)
    for i in result:
        print(i,end=" ")
