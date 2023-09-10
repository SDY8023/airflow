# coding=utf-8
# @Author:SDY
# @File:test28.py
# @Time:2023/3/23 20:51
# @Introduction:
import re
if __name__ == '__main__':
    ip = input()
    pattern = r"[0-9]"
    d = re.findall(pattern,ip)
    datas = ip.split(".")
    result = "YES"
    if len(datas) != 4:
        result = "NO"
    else:
        for d in datas:
            if len(re.findall(pattern, d)) != len(d) or d == "" or (d.startswith("0") and len(d) != 1):
                result = "NO"
                break
            else:
                s = bin(int(d))
                if len(s) > 10:
                    result = "NO"
                    break
    print(result)