import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 读取数据,默认从第三行开始读取
def readData(filePath,start_row=3):
    data = pd.read_excel(filePath,skiprows=range(start_row))
    head_data = data.head()
    print(head_data)
    return data

# 画图
def draw(data):
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 9))
    print(data[['PQI']])
    plt.plot(data[['PQI']])
    plt.xticks(range(0, data.shape[0], 20), data['PQI'], rotation=45)
    plt.title("****** Stock Price", fontsize=18, fontweight='bold')
    plt.xlabel('年份', fontsize=18)
    plt.ylabel('PQI', fontsize=18)
    plt.show()


if __name__ == '__main__':
    data = readData("F:\Study\项目\道路项目\数据.xlsx",1)
    draw(data)

