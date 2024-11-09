import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 设置中文字体
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取训练数据
data = pd.DataFrame({
    '客货比': [0.457349, 0.499813, 0.575952, 0.600000, 0.666982, 0.657092, 0.711176, 0.688291, 0.676264, 0.666794],
    '温度': [12.65, 12.75, 13.17, 13.42, 14.46, 13.96, 14.21, 14.09, 13.83, 14.25],
    '路龄': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    '交通量': [22804.41, 22723.40, 19286.29, 19154.37, 22708.10, 23863.15, 24446.79, 28908.72, 33188.92, 37112.67],
    'PQI': [90.99397, 86.31145, 88.25088, 95.13, 96.0, 95.2, 90.3, 93.4, 94.9, 92.09]
})

time_steps = 3  # 设置时间步长

# 数据标准化
scaler = StandardScaler()
scaler_target = StandardScaler()
features = scaler.fit_transform(data[['客货比', '温度', '路龄', '交通量']].values)
target = scaler_target.fit_transform(data[['PQI']].values)

# 构建输入序列
x, y = [], []
for i in range(len(features) - time_steps + 1):
    x.append(features[i:i + time_steps])
    y.append(target[i + time_steps - 1])

x = np.array(x)
y = np.array(y)


# 转换为PyTorch 张量
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


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


# 模型初始化
input_size = x.shape[2]
hidden_size = 50
output_size = y.shape[1]
num_layers = 1
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 200
loss_values = []
for epoch in range(num_epochs):
    model.train()
    outputs = model(x)
    optimizer.zero_grad()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'训练轮次 [{epoch + 1}/{num_epochs}], 损失: {loss.item():.4f}')


# 预测训练集上的 PQI
model.eval()
with torch.no_grad():
    predicted = model(x).cpu().numpy()

# 逆标准化预测结果和目标
predicted = scaler_target.inverse_transform(predicted)
y = scaler_target.inverse_transform(y)

# 打印预测结果和实际值
print("Predicted:")
print(predicted)

print("Actual:")
print(y)

# 绘制预测结果与实际值对比图
plt.figure(figsize=(10, 5))
plt.plot(predicted, label='预测PQI', marker='o')
plt.plot(y, label='实际PQI', marker='x')
plt.xlabel('样本索引')
plt.ylabel('PQI值')
plt.title('预测值与实际值对比')
plt.legend()
plt.savefig('predicted_vs_actual.png')
#plt.show()

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_values)
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title('训练损失随训练轮次变化曲线')
plt.savefig('training_loss.png')
#plt.show()

# 读取预测数据
predict_data = pd.DataFrame({
    '客货比': [0.6, 0.666982, 0.657092, 0.711176, 0.688291, 0.676264, 0.666794],
    '温度': [13.42, 14.46, 13.96, 14.21, 14.09, 13.83, 14.25],
    '路龄': [4, 5, 6, 7, 8, 9, 10],
    '交通量': [19154.37, 22708.10, 23863.15, 24446.79, 28908.72, 33188.92, 37112.67]
})

# 数据标准化
predict_features = scaler.transform(predict_data[['客货比', '温度', '路龄', '交通量']].values)


# 独立预测每一组数据
predicted_new = []
with torch.no_grad():
    for i in range(len(predict_features)):
        input_seq = predict_features[i:i + 1]
        input_seq = np.expand_dims(input_seq, axis=0)  # 添加 batch 维度和时间步维度
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        pred = model(input_tensor).cpu().numpy()
        predicted_new.append(pred)


predicted_new = np.concatenate(predicted_new, axis=0)

# 逆标准化预测结果
predicted_new = scaler_target.inverse_transform(predicted_new)


# 打印预测数据和预测结果
for i in range(len(predict_data)):
    print("预测数据:")
    for column, value in predict_data.iloc[i].items():
        print(f"{column}: {value}")
    print(f"预测结果: {predicted_new[i][0]}")
    print()
