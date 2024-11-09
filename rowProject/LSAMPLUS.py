# coding=utf-8
# @Author:SDY
# @File:LSAMPLUS.py
# @Time:2024/6/24 21:34
# @Introduction:
import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# 设置中文字体
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# 配置日志记录器
logging.basicConfig(filename=f'logs/app_log_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt', filemode='w',format='%(name)s - %(levelname)s - %(message)s')
# 设置日志级别，低于该级别的日志不会被记录
logging.getLogger().setLevel(logging.INFO)
# 客货比
feature1 = [0.457349, 0.499813, 0.575952, 0.600000, 0.666982, 0.657092, 0.711176, 0.688291, 0.676264, 0.666794]
# 温度
feature2 = [12.65, 12.75, 13.17, 13.42, 14.46, 13.96, 14.21, 14.09, 13.83, 14.25]
# 路龄
feature3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 交通量
feature4 = [22804.41, 22723.40, 19286.29, 19154.37, 22708.10, 23863.15, 24446.79, 28908.72, 33188.92, 37112.67]
# PQI
target = [90.99397, 86.31145, 88.25088, 95.13, 96.0, 95.2, 90.3, 93.4, 94.9, 92.09]

"""
构建数据
"""
def dealData(feature1,feature2,feature3,feature4,target,path,num_epochs):
    logging.info(f"============开始预测 {len(target)} 年PQI============")
    # 前n-1 组数据为 历史数据
    historyData = pd.DataFrame({
        '客货比': feature1,
        '温度': feature2,
        '路龄': feature3,
        '交通量': feature4,
        'PQI': target
    })

    scaler = StandardScaler()
    scaler_target = StandardScaler()
    features = scaler.fit_transform(historyData[['客货比', '温度', '路龄', '交通量']].values)
    target_train = scaler_target.fit_transform(historyData[['PQI']].values)
    # 构建输入序列
    x, y  = [], []
    for i in range(len(features)):
        x.append(features[i:i+1])
        y.append(target_train[i])
    x = np.array(x)
    y = np.array(y)
    # 处理预测数据
    # for i in range(len(currentFeatures)):
    #     c.append()

    # 转换为PyTorch 张量
    t_x = torch.tensor(x[0:len(x)-1], dtype=torch.float32)
    t_y = torch.tensor(y[0:len(y)-1], dtype=torch.float32)
    t_c = torch.tensor(x[len(x)-1:len(x)],dtype=torch.float32)
    # 模型初始化
    # 这个地方的作用？
    input_size = x.shape[2]
    hidden_size = 50
    output_size = y.shape[1]
    num_layers = 1
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    # 定义损失函数和优化器
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 训练模型
    #num_epochs = 200
    loss_values = []
    for epoch in range(num_epochs):
        model.train()
        outputs = model(t_x)
        optimizer.zero_grad()
        loss = criterion(outputs, t_y)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

        if (epoch + 1) % 10 == 0:
            logging.info(f'训练轮次 [{epoch + 1}/{num_epochs}], 损失: {loss.item():.4f}')
    # 画每次预测损失图
    logging.info(f"====绘制每次预测损失图")
    fileName = f"预测{len(target)}年训练损失变化曲线"
    drawLossChart(loss_values,fileName,path)
    # 预测训练集上的 PQI
    model.eval()
    with torch.no_grad():
        predicted = model(t_c).cpu().numpy()
    # 逆标准化预测结果和目标,把数据转换为实际值
    predicted = scaler_target.inverse_transform(predicted)
    # actual = scaler_target.inverse_transform(t_y)

    logging.info(f"============结束预测 {len(target)} 年PQI 预测PQI:{predicted[-1][-1]}============")
    return predicted[-1]

def drawChart(t_y,predicted,fileName,path):

    # 绘制预测结果与实际值对比图
    plt.figure(figsize=(20, 5))
    plt.plot(predicted, label='预测PQI', marker='o')
    plt.plot(t_y, label='实际PQI', marker='x')
    plt.xlabel('样本索引')
    plt.ylabel('PQI值')
    plt.title('预测值与实际值对比')
    plt.legend()
    plt.savefig(f'{path}/{fileName}.png')
    plt.close()
    # plt.show()

def drawLossChart(loss_values,fileName,path):
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values)
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('训练损失随训练轮次变化曲线')
    plt.savefig(f'{path}/{fileName}.png')
    plt.close()
    # plt.show()

def makeDir(pathName):
    if not os.path.exists(pathName):
        os.makedirs(pathName)
        logging.info(f"目录 {pathName} 已创建")
    else:
        logging.info(f"目录 {pathName} 已存在")


# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def buildData(year,data_size):
    use_feature1 = []
    use_feature2 = []
    use_feature3 = []
    use_feature4 = []
    use_target = []
    if(year <= data_size):
        use_feature1 = feature1[0:year]
        use_feature2 = feature2[0:year]
        use_feature3 = feature3[0:year]
        use_feature4 = feature4[0:year]
        use_target = target[0:year]
    else:
        quotient = year // data_size
        remainder = year % data_size
        for i in range(quotient):
            use_feature1 += feature1
            use_feature2 += feature2
            use_feature4 += feature4
            use_target += target
        use_feature1 = use_feature1 + feature1[0:remainder]
        use_feature2 = use_feature2 + feature2[0:remainder]
        for i in range(year):
            use_feature3.append(i+1)
        use_feature4 = use_feature4 + feature4[0:remainder]
        use_target = use_target + target[0:remainder]

    return use_feature1,use_feature2,use_feature3,use_feature4,use_target

def predictedData(user_input,num_epochs,flag):
    # 预测数据集
    predictedList = []
    targetList = []
    # 创建路径
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = f"picture/训练{num_epochs}次-预测{user_input}年{current_time}"
    makeDir(path)
    build_feature1, \
    build_feature2, \
    build_feature3, \
    build_feature4, \
    build_feature5 = buildData(user_input, len(feature1))
    if(flag.lower() == "yes"):
        for i in range(0, user_input):
            print(f"预测{i}")
            preictedResult = dealData(build_feature1[0:i + 1],
                                      build_feature2[0:i + 1],
                                      build_feature3[0:i + 1],
                                      build_feature4[0:i + 1],
                                      build_feature5[0:i + 1], path, num_epochs)
            predictedList.append([preictedResult[-1]])
            targetList.append(build_feature5[i:i + 1])
    else:
        preictedResult = dealData(build_feature1,
                                  build_feature2,
                                  build_feature3,
                                  build_feature4,
                                  build_feature5, path, num_epochs)
        print(f"预测PQI值:{preictedResult[-1]}")
        print(f"实际PQI值:{build_feature5[-1]}")



    logging.info("本次预测值集合")
    logging.info(predictedList)
    logging.info("本次预测实际值集合")
    logging.info(targetList)
    drawChart(targetList,predictedList,"目标值PQI-VS-预测PQI",path)
if __name__ == '__main__':
    user_input = int(input("请输入需要预测的年份:"))
    num_epochs = int(input("请输入需要训练的次数:"))
    flag = input("是否需要迭代预测:")
    predictedData(user_input,num_epochs,flag)








