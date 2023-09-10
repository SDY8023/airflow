# coding=utf-8
# @Author:SDY
# @File:test22.py
# @Time:2023/3/12 12:12
# @Introduction:
if __name__ == '__main__':
    n = int(input())
    result = {}
    for i in range(n):
        k,v = map(int,input().split())
        if k in result:
            result[k] = result[k] + v
        else:
            result[k] = v
    # dict集合的排序方法
    for d in sorted(result):
        print("{} {}".format(d,result[d]))