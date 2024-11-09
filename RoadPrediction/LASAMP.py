# coding=utf-8
# @Author:SDY
# @File:LSAMPLUS.py
# @Time:2024/08/04 21:34
# @Introduction:
import tkinter
import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
# 设置中文字体
from matplotlib import rcParams
plt.rcParams['font.family'] = ['SimHei']  # 例如使用黑体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 确保使用黑体

class RoadPredictionParam:
    # 训练次数
    num_epochs = '200'
    # 以最后多少年数据为测试集
    test_row = '5'
    # 预测多少年
    prediction_years = '1'
    hidden_size = '50'
    num_layers = '1'
    reduction = 'sum'
    lr = 0.01
    aa = 10
    def __init__(self):
        pass


class RoadPrediction:
    log_text = None

    def __init__(self):
        pass

    def dealData(self,feature1, feature2, feature3, feature4, target, path, analysis_param:RoadPredictionParam):
        self.log_text.insert(tk.END, f"=====开始预测=====\n")
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
        x, y = [], []
        for i in range(len(features)):
            x.append(features[i:i + 1])
            y.append(target_train[i])
        x = np.array(x)
        y = np.array(y)
        # 处理预测数据
        # for i in range(len(currentFeatures)):
        #     c.append()

        # 转换为PyTorch 张量
        t_x = torch.tensor(x[0:len(x) - int(analysis_param.test_row)], dtype=torch.float32)
        t_y = torch.tensor(y[0:len(y) - int(analysis_param.test_row)], dtype=torch.float32)
        # 最后n组数据作为测试集
        t_c = torch.tensor(x[len(x) - int(analysis_param.test_row):len(x)], dtype=torch.float32)
        # 模型初始化
        # 这个地方的作用？
        input_size = x.shape[2]
        hidden_size = int(analysis_param.hidden_size)
        output_size = y.shape[1]
        num_layers = int(analysis_param.num_layers)
        model = LSTMModel(input_size, hidden_size, output_size, num_layers)
        # 定义损失函数和优化器
        criterion = nn.MSELoss(reduction=analysis_param.reduction)
        optimizer = optim.Adam(model.parameters(), lr=float(analysis_param.lr))
        # 训练模型
        # num_epochs = 200
        loss_values = []
        for epoch in range(int(analysis_param.num_epochs)):
            model.train()
            outputs = model(t_x)
            optimizer.zero_grad()
            loss = criterion(outputs, t_y)
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

            if (epoch + 1) % 10 == 0:
                self.log_text.insert(tk.END, f'训练轮次 [{epoch + 1}/{analysis_param.num_epochs}], 损失: {loss.item():.4f}\n')
        # 画每次预测损失图
        self.log_text.insert(tk.END, f'====绘制每次预测损失图\n')
        fileName = f"预测{len(target)}年训练损失变化曲线"
        self.drawLossChart(loss_values, fileName, path)
        # 预测测试集的 PQI
        model.eval()
        with torch.no_grad():
            predicted_test = model(t_c).cpu().numpy()
        # 训练集预测值
        with torch.no_grad():
            predicted_x = model(t_x).cpu().numpy()
        # 逆标准化预测结果和目标,把数据转换为实际值
        predicted = scaler_target.inverse_transform(predicted_test)
        # 测试集预测
        predicted_actual = scaler_target.inverse_transform(predicted_x)
        # actual = scaler_target.inverse_transform(t_y)

        logging.info(f"============结束预测 {len(target)} 年PQI 预测PQI:{predicted[-1][-1]}============")
        self.log_text.insert(tk.END, f"============结束预测 {len(target)} 年PQI 预测PQI:{predicted[-1][-1]}============\n")
        return (predicted_actual, predicted,loss_values)

    def drawChart(self,t_y, predicted, fileName, path):

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
        self.log_text.insert(tk.END, f"目标值PQI-VS-预测PQI 预测图存储位置:{path}/{fileName}.png\n")
        # plt.show()

    def drawLossChart(self,loss_values, fileName, path):
        # 绘制训练损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values)
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title('训练损失随训练轮次变化曲线')
        plt.savefig(f'{path}/{fileName}.png')
        plt.close()
        # plt.show()

    def drawResultChart(self,result, fileName, path):
        fig = go.Figure()
        # 添加训练预测轨迹
        fig.add_trace(go.Scatter(x=result['year'], y=result["predict"], mode='lines', name='Train prediction'))
        # 添加测试预测轨迹
        fig.add_trace(go.Scatter(x=result['year'], y=result["test"], mode='lines', name='Test prediction'))
        # 添加实际值轨迹
        fig.add_trace(go.Scatter(x=result['year'], y=result["target"], mode='lines', name='Actual Value'))
        fig.update_layout(
            xaxis=dict(title_text='Date', dtick=1, showline=True, showgrid=True, showticklabels=True, linecolor='white',
                       linewidth=2),
            yaxis=dict(title_text='PQI', dtick=0.2, titlefont=dict(family='Rockwell', size=12, color='white', ),
                       showline=True,
                       showgrid=True, showticklabels=True, linecolor='white', linewidth=2, ticks='outside',
                       tickfont=dict(family='Rockwell', size=12, color='white', ), ),
            showlegend=True, template='plotly_dark')

        annotations = []
        annotations.append(dict(
            xref='paper', yref='paper', x=0.0, y=1.05,
            xanchor='left', yanchor='bottom',
            text='预测PQI (LSTM)',
            font=dict(family='Rockwell', size=26, color='white'),
            showarrow=False
        ))
        fig.update_layout(annotations=annotations)
        fig.show()

    def makeDir(self,pathName):
        if not os.path.exists(pathName):
            os.makedirs(pathName)
            logging.info(f"目录 {pathName} 已创建")
        else:
            logging.info(f"目录 {pathName} 已存在")

    def predictedData(self, column_lists,log_text,analysis_param:RoadPredictionParam):
        self.log_text = log_text
        # 创建路径
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = f"picture/训练{analysis_param.num_epochs}次-预测年{current_time}"
        self.makeDir(path)
        build_feature1 = column_lists["客货比"]
        build_feature2 = column_lists["温度"]
        build_feature3 = column_lists["年份"]
        build_feature4 = column_lists["交通量"]
        build_feature5 = column_lists["PQI"]
        preictedResult = self.dealData(build_feature1,
                                       build_feature2,
                                       build_feature3,
                                       build_feature4,
                                       build_feature5, path, analysis_param)
        predictedList = preictedResult[0]
        testList = preictedResult[1]
        loss_values = preictedResult[2]
        targetList = build_feature5
        predictDataList = []
        testDataList = []
        finalResultList = []
        for i in range(len(predictedList)):
            predictDataList.append(predictedList[i][0])
            finalResultList.append(predictedList[i][0])
        testDataList.append(predictDataList[-1])
        for i in range(len(testList)):
            testDataList.append(testList[i][0])
            finalResultList.append(testList[i][0])
        log_text.insert(tk.END, f"本次训练集预测值集合:{predictDataList}\n")
        log_text.insert(tk.END, f"本次测试集预测值集合:{testDataList}\n")
        log_text.insert(tk.END, f"本次实际值集合:{targetList}\n")
        predictedListResult = pd.Series(predictDataList).reindex(range(len(targetList)))
        result = pd.DataFrame({
            "predict": predictedListResult,
            "target": targetList,
            "year": build_feature3
        })
        # 创建一个与目标长度一致的全NaN的Series
        test_series = pd.Series(np.nan, index=range(len(targetList)))
        # 将testList的数据填充到最后几位
        test_series[-len(testDataList):] = testDataList
        result["test"] = test_series
        log_text.insert(tk.END, f"开始画预测图\n")
        #self.drawResultChart(result, "目标值PQI-VS-预测PQI", path)
        self.drawChart(targetList, finalResultList, "目标值PQI-VS-预测PQI", path)
        return (loss_values,targetList,finalResultList)

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








