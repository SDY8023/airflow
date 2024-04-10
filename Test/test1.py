# coding=utf-8
# @Author:SDY
# @File:test1.py
# @Time:2023/10/29 10:25
# @Introduction:
import json
if __name__ == '__main__':
    readFilePath = "F:\Study\spark\spark3.0优化\资料\\1.数据文件\courseshoppingcart.log"
    writeFilePath = "F:\Study\spark\spark3.0优化\资料\\1.数据文件\courseshoppingcart.txt"
    with open(readFilePath,mode="r") as rf:
        while True:
            line = rf.readline()
            data = json.loads(line)
            # print(line)
            writeData = str(data["courseid"])+"|"+data["coursename"]+"|"+data["createtime"]+"|"+data["discount"]+"|"+data["dn"]+"|"+data["dt"]+"|"+data["orderid"]+"|"+data["sellmoney"]
            # print(writeData)
            with open(writeFilePath,"a+") as wf:
                wf.write(writeData+"\n")

