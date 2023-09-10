# coding=utf-8
# @Author:SDY
# @File:test27.py
# @Time:2023/3/15 21:53
# @Introduction:
import re
def check_length(p):
    score = 0
    l = len(p)
    if l <= 4:
        score = 5
    elif l <= 7:
        score = 10
    else:
        score = 25
    return score

def check_number(p):
    score = 0
    n = re.findall(r"[0-9]",p)
    if len(n) == 1:
        score = 10
    elif len(n) > 1:
        score = 20
    else:
        score = 0
    return score

def check_AZ(p):
    score = 0
    az = re.findall(r"[a-z]",p)
    AZ = re.findall(r"[A-Z]",p)
    if len(az) == 0 and len(AZ) == 0:
        score = 0
    elif len(az) != 0 and len(AZ) == 0:
        score = 10
    elif len(az) == 0 and len(AZ) != 0:
        score = 10
    else:
        score = 20
    return score

def check_ascii(p):
    score = 0
    asc = re.findall(r'[!"#\$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]',p)
    if len(asc) == 0:
        score = 0
    elif len(asc) == 1:
        score = 10
    else:
        score = 25
    return score


if __name__ == '__main__':
    p = input()
    len_score = check_length(p)
    number_score = check_number(p)
    AZ_score = check_AZ(p)
    asc_score = check_ascii(p)
    total_score = len_score + AZ_score + asc_score + number_score
    max_score = 0
    # 只有字母和数字
    if AZ_score == 10 and number_score != 0:
        max_score = 2
    if AZ_score == 10 and number_score != 0 and asc_score != 0:
        max_score = 3
    if AZ_score == 20 and number_score != 0 and asc_score != 0:
        max_score = 5
    total_score += max_score
    print(total_score)
    if total_score >= 90:
        print("VERY_SECURE")
    elif total_score >= 80:
        print("SECURE")
    elif total_score >= 70:
        print("VERY_STRONG")
    elif total_score >= 60:
        print("STRONG")
    elif total_score >= 50:
        print("AVERAGE")
    elif total_score >= 25:
        print("WEAK")
    else:
        print("VERY_WEAK")
