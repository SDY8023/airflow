# coding=utf-8
# @Author:SDY
# @File:test13.py
# @Time:2023/3/11 22:06
# @Introduction:
while True:
    try:
        m, n = map(int, input().split())
        x1, y1, x2, y2=map(int,input().split())
        insert_x=int(input())
        insert_y=int(input())
        x,y=map(int,input().split())
        if (0 <= m <= 9) and (0 <= n <= 9):
            print('0')
        else:
            print('-1')
        if (0 <= x1 < m) and (0 <= y1 < n) and (0 <= x2 <= m)and (0 <= y2 < n):
            print('0')
        else:
            print('-1')
        if (0 <= insert_x < m) and (m < 9):
            print('0')
        else:
            print('-1')
        if (0 <= insert_y < n) and (n < 9):
            print('0')
        else:
            print('-1')
        if(0 <= x < m)and (0 <= y < n):
            print('0')
        else:
            print('-1')
    except:
        break


