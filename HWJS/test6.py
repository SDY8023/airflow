# coding=utf-8
# @Author:SDY
# @File:test6.py
# @Time:2023/2/26 14:59
# @Introduction:
money = input()
count = 100
for i in range(0,101):
    total_money = 100
    # 鸡翁的数量
    if i * 5 < total_money:
        total_money = total_money - i * 5
        for j in range(0,101):
            # 鸡母的数量
            total_money_2 = total_money
            if j * 3 < total_money_2:
                total_money_2 = total_money_2 - j * 3
                # 鸡雏的数量
                m = total_money_2 * 3
                if i + j + m == 100:
                    print(i,j,m)

