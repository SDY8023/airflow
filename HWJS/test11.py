# coding=utf-8
# @Author:SDY
# @File:test11.py
# @Time:2023/3/11 19:43
# @Introduction:

def getAllEgyptGrade(data):
    datas = data.split("/")
    fz = int(datas[0])
    fm = int(datas[1])
    result = []
    while fz != 1:
        d = getMaxRgyptGrade(fz,fm)
        result.append("1/{}".format(d))
        fz = fz * d-fm
        fm = fm * d
        new_fz_fm = reduction(fz,fm)
        fz = new_fz_fm[0]
        fm = new_fz_fm[1]
    result.append("{}/{}".format(fz,fm))
    return result

'''
辗转相除法
'''
def reduction(fz,fm):
    tmp = -1
    d1 = fz
    d2 = fm
    result_reduction = []
    max_reduction = 0
    while tmp != 0:
        consult_remainder = divmod(d2,d1)
        d2 = d1
        d1 = consult_remainder[1]
        tmp = consult_remainder[1]
        if tmp != 0:
            result_reduction.append(consult_remainder[1])
    if len(result_reduction) != 0:
        result_reduction.sort()
        max_reduction = result_reduction[0]
    if max_reduction != 0:
        return ((int(fz/max_reduction),int(fm/max_reduction)))
    else:
        return tuple((1,int(fm/fz)))


def getMaxRgyptGrade(fz,fm):
    consult_remainder = divmod(fm,fz)
    # 商
    consult = consult_remainder[0]
    return consult + 1


if __name__ == '__main__':
    data = input()
    all_egypt_grade = getAllEgyptGrade(data)
    for i in range(len(all_egypt_grade)):
        if i != len(all_egypt_grade)-1:
            print('{}+'.format(all_egypt_grade[i]),end='')
        else:
            print(all_egypt_grade[i],end='')
