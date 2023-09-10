# coding=utf-8
# @Author:SDY
# @File:test25.py
# @Time:2023/3/12 14:43
# @Introduction:
if __name__ == '__main__':
    # 第一个矩阵行数
    x = int(input())
    # 第一个矩阵列数，第二个矩阵行数
    y = int(input())
    # 第二个矩阵列数
    z = int(input())
    data1 = []
    data2 = []
    result = []
    # 第一个矩阵的值
    for i in range(x):
        d1 = input().split(" ")
        data1.append(list(d1))
    # 第二个矩阵的值
    for i in range(y):
        d2 = input().split(" ")
        data2.append(list(d2))
     # 计算
    for i in range(x):
        tmp_list = []
        for n in range(z):
            tmp_data = 0
            for j in range(y):
                tmp_data += int(data1[i][j]) * int(data2[j][n])
            tmp_list.append(tmp_data)
        result.append(tmp_list)
    for d in result:
        for e in d:
            print(e,end=" ")
        print()