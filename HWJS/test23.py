# coding=utf-8
# @Author:SDY
# @File:test23.py
# @Time:2023/3/12 12:31
# @Introduction:
if __name__ == '__main__':
    n = input()
    result = []
    for i in range(len(n)):
        if n[len(n)-i-1] not in result:
            result.append(n[len(n)-i-1])
            print(n[len(n)-i-1],end="")