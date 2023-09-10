# coding=utf-8
# @Author:SDY
# @File:test1.py
# @Time:2023/4/9 16:11
# @Introduction:

import random
def getFrontNumber():
    # 生成前区号码(5个不重复的1到35之间的数字)
    front_numbers = set()
    while len(front_numbers) < 5:
        front_numbers.add(random.randint(1, 35))
    front_numbers = sorted(list(front_numbers))
    result_numbers = list()
    for d in front_numbers:
        n = str(d)
        if len(n) < 2:
            n = "0" + n
        result_numbers.append(n)

    return result_numbers

def getBackNumber():
    # 生成后区号码(2个不重复的1到12之间的数字)
    back_numbers = set()
    while len(back_numbers) < 2:
        back_numbers.add(random.randint(1, 12))
    back_numbers = sorted(list(back_numbers))
    result_numbers = list()
    for d in back_numbers:
        n = str(d)
        if len(n) < 2:
            n = "0" + n
        result_numbers.append(n)

    return result_numbers

def daLeTou(n):
    # 获取中奖号码
    win_front_number = getFrontNumber()
    win_back_number = getBackNumber()
    print("一等奖号码 ===> {}|{}".format(" ".join(win_front_number)," ".join(win_back_number)))
    # 生成机选的号码
    flag = True
    i = 1
    while flag:
        person_front_numbers = getFrontNumber()
        person_back_numbers = getBackNumber()
        print("第{}次机选号码 ===> {}|{}".format(str(i)," ".join(person_front_numbers)," ".join(person_back_numbers)))
        person_win_front_list = set(win_front_number).intersection(set(person_front_numbers))
        person_win_back_list = set(win_back_number).intersection(set(person_back_numbers))
        if len(person_win_front_list) == 5 and len(person_win_back_list) == 2:
            flag = False
            print("恭喜中了1等奖 10000000元！")
        else:
            i += 1


if __name__ == '__main__':
    daLeTou(5)
