# coding=utf-8
# @Author:SDY
# @File:writeFile.py
# @Time:2023/6/3 15:52
# @Introduction:

import random
import string
file_path = 'D:\\study\\testData\\10gbFIle.txt'
line_length = 1024  # 每行长度为1KB
num_lines = 10 * 1024 * 1024  # 总行数
# 生成随机内容
def generate_random_content(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

# 生成文件
with open(file_path, 'w') as file:
    for _ in range(num_lines):
        line_content = generate_random_content(line_length)
        file.write(line_content)
        file.write('\n')