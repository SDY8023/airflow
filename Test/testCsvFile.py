# coding=utf-8
# @Author:SDY
# @File:testCsvFile.py
# @Time:2023/10/7 22:00
# @Introduction:
import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv('F:\\Study\\creatData\\01FT20230818001_93K12_BTV_KF1968E_K2NA351-2332_FT1_RT1_20230826093706_KF1968_FTALL1_V93KS4_V02_NV004.csv',low_memory=False)
    #df = pd.read_csv('F:\\Study\\creatData\\test1.csv')
    #print(df["(C)PTR_990 :Temp_Test:Env_Judge[1]@"])
    print(df.head())
