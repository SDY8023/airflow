# coding=utf-8
# @Author:SDY
# @File:test12.py
# @Time:2023/3/11 21:20
# @Introduction:

def change_cell(line,column,x1,y1,x2,y2):
    result = -2
    if x1 > line -1 or x2 > line -1 or y1 > column-1 or y2 > column-1:
        result = -1
    else:
        result = 0
    return result

def insert_line_column(line,column,insert,type):
    result = -2
    if type == 0 and line + 1 <= 9 and 0 <= insert < line :
        result = 0
    elif type == 1 and column + 1 <= 9 and 0 <= insert < column:
        result = 0
    else:
        result = -1
    return result

def query(line,column,x,y):
    result = -2
    if x > line-1 or y > column-1:
        result = -1
    else:
        result = 0
    return result

def check_line_column(line,column):
    result = -2
    if line > 9 or column > 9:
        result = -1
    else:
        result = 0
    return result


if __name__ == '__main__':
    while True:
        try:
            line,column = map(int,input().split())
            # 交换单元格
            x1, y1, x2, y2=map(int,input().split())
            # 插入行
            insert_line = int(input())
            # 插入列
            insert_column = int(input())
            # 查询
            x,y = map(int,input().split())
            print(check_line_column(line, column))
            print(change_cell(line, column, x1,y1,x2,y2))
            print(insert_line_column(line, column, insert_line, 0))
            print(insert_line_column(line, column, insert_column, 1))
            print(query(line, column, x,y))
        except:
            break

