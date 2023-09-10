# coding=utf-8
# @Author:SDY
# @File:test7.py
# @Time:2023/2/26 15:27
# @Introduction:
date = input()
dates = date.split(' ')
year = int(dates[0])
month = int(dates[1])
day = int(dates[2])

month_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]

if year % 4 == 0 and year % 100 != 0:
    month_days[2] = month_days[2] + 1
total_days = 0
for i in range(1,month):
    total_days += month_days[i]
total_days += day
print(total_days)