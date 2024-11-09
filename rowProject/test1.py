# coding=utf-8
# @Author:SDY
# @File:LSAMPLUS.py
# @Time:2024/6/24 21:34
# @Introduction:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import chardet
filepath = 'F:\Study\项目\道路项目\数据.xlsx'
data = pd.read_excel(filepath)

sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(data['PQI'])
plt.xticks(range(0,data.shape[0],20), data['Date'].loc[::20], rotation=45)
plt.title("****** Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('PQI',fontsize=18)
plt.show()

# 选取PQI作为特征
price = data[['PQI']]
# 数据归一化，将数据放缩到-1和1之间
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
price['PQI'] = scaler.fit_transform(price['PQI'].values.reshape(-1,1))

def split_data(stock, lookback):
    data_raw = stock.to_numpy()
    data = []
    # 按照8:2的比例划分训练集和测试集
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    return [x_train, y_train, x_test, y_test]
# 利用前19天的数据预测第20天的数据
lookback = 20
x_train, y_train, x_test, y_test = split_data(price, lookback)
# 打印训练集和测试集维度
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


x_train = torch.Tensor(x_train).type(torch.Tensor)
x_test = torch.Tensor(x_test).type(torch.Tensor)
y_train_lstm = torch.Tensor(y_train).type(torch.Tensor)
y_test_lstm = torch.Tensor(y_test).type(torch.Tensor)
y_train_gru = torch.Tensor(y_train).type(torch.Tensor)
y_test_gru = torch.Tensor(y_test).type(torch.Tensor)
input_dim = 1    # 输入特征的维数
hidden_dim = 50  # 隐藏层的神经元个数
num_layers = 3   # 隐藏层的层数
output_dim = 1   # 输出特征的维数
num_epochs = 100   # 网络迭代次数
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss() # 使用均方差作为损失函数
optimiser = torch.optim.Adam(model.parameters(), lr=0.01) # 使用Adam优化算法

import time
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
import seaborn as sns
sns.set_style("darkgrid")
fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)
plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
ax.set_title('Stock price', size = 14, fontweight='bold')
ax.set_xlabel("Date", size = 14)
ax.set_ylabel("Clost", size = 14)
ax.set_xticklabels('', size=10)
plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)

import math, time
from sklearn.metrics import mean_squared_error
y_test_pred = model(x_test)
# 将数据还原成放缩前的大小
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())
# 训练集数据的RMSE
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
# 测试集数据的RMSE
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
lstm.append(trainScore)
lstm.append(testScore)
lstm.append(training_time)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred
# shift test predictions for plotting
testPredictPlot = np.empty_like(price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

original = scaler.inverse_transform(price['PQI'].values.reshape(-1,1))
print("trainPredictPlot")
print(trainPredictPlot)
predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)

import plotly.express as px
import plotly.graph_objects as go
print("result.index")
print(result.index)
print("result[0]")
print(result[0].__class__)
print("result")
print(result)
fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],mode='lines',name='Train prediction')))
fig.add_trace(go.Scatter(x=result.index, y=result[1], mode='lines',name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],mode='lines',name='Actual Value')))
fig.update_layout(
    xaxis=dict(title_text='Date',showline=True,showgrid=True,showticklabels=False,linecolor='white',linewidth=2),
    yaxis=dict(title_text='PQI',titlefont=dict(family='Rockwell',size=12,color='white',),showline=True,showgrid=True,showticklabels=True,linecolor='white',linewidth=2,ticks='outside',tickfont=dict(family='Rockwell',size=12,color='white',),),
    showlegend=True,template = 'plotly_dark')
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,xanchor='left', yanchor='bottom',text='Results (LSTM)',font=dict(family='Rockwell',size=26,color='white'),showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()
