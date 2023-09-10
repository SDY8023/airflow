# coding=utf-8
# @Author:SDY
# @File:test24.py
# @Time:2023/3/12 14:28
# @Introduction:
if __name__ == '__main__':
    while True:
        try:
            s = input()  # 输入
            len_s = len(s)
            result = 0  # result用于记录目前最长的子串长度，初始化为0
            for n in range(len_s):  # 从第一个字符的位置开始检查
                max_len = result + 1  # max_len代表当前检查字符串的长度
                while n + max_len <= len_s:  # 代表搜索完以第n+1个字符串开头的所有字符串
                    if s[n:n + max_len] == s[n:n + max_len][::-1]:  # 如果满足是回文字符串
                        result = max_len  # 则此时最大的字符串长度变为max_len
                    max_len += 1  # 每次增加字符串的长度1
            if result != 0:
                print(result)
        except:
            break

