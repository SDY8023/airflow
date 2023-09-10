# coding=utf-8
# @Author:SDY
# @File:test8.py
# @Time:2023/2/26 15:47
# @Introduction:
str = input()
datas = str.split(' ')
result = []
tmp_str = ''
flag = 0
for d in datas:
    if (d.startswith('"') or d.endswith('"')) and not (d.startswith('"') and d.endswith('"')) or flag == 1:
        if flag == 0:
            tmp_str += d.replace('"','')
            flag += 1
        elif flag == 1 and d.endswith('"'):
            tmp_str = '{} {}'.format(tmp_str,d.replace('"',''))
            result.append(tmp_str)
            tmp_str = ''
            flag = 0
        elif flag == 1 and not d.endswith('"'):
            tmp_str = '{} {}'.format(tmp_str,d)
    elif d.startswith('"') and d.endswith('"'):
        result.append(d.replace('"',''))
    else:
        result.append(d)
print(len(result))
for d in result:
    print(d)