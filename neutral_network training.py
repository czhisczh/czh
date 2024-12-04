import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, file):
        self.data = self.load_data(file)

    def load_data(self, file):
        data = pd.read_csv(file)
        return data

    def __len__(self):
        return len(self.data)

#__getitem__方法的作用就是从dataframe中提取数据并转化为张量
    def __getitem__(self, idx):
        if idx >= 3:
            x = self.data.iloc[idx - 3:idx, 1].to_numpy()
            y = self.data.iloc[idx, 1]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        #torch.tensor()是常见张量返回方法，可接受列表、numpy数组等数据类型
        #dtype=torch.float32 32位浮动数的数据类型有助于提高计算效率并节省内存
        #to_numpy()将 Pandas 的 DataFrame 或 Series 转换为 NumPy 数组
        else:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 假设输入特征是10维
        self.fc2 = nn.Linear(64, 1)   # 输出1维 (回归)
#forward()前向传播函数主要定义，数据如何通过神经网络流动
#relu函数课堂有讲，将输入中小于0的部分置为0，其他不变
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 设备设置(用CPU算或者GPU算，用GPU要下载CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file = "youtube_play_counts.csv"  # 数据文件路径
dataset = MyDataset(file)
#这里通过MyDataset类中的load_data方法加载了csv数据，存储位置是self.data，格式是Dataframe
#dataset是类实例化的对象，作为一个对象来提供数据，一个batch一个batch地给，这个对象的各种数据、属性在第一个类中定义了

tr_set = DataLoader(dataset, batch_size=16, shuffle=True)  # 训练集
dv_set = DataLoader(dataset, batch_size=16, shuffle=False) # 验证集
tt_set = DataLoader(dataset, batch_size=16, shuffle=False) # 测试集
#Dataloader会自动调用__getitem__方法提取数据，并且打包成一个批次
#关于数据类型：dataset在此阶段（实例化）还是一个包含dataframe的对象


# 模型构建
model = MyModel().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练和验证循环
n_epochs = 20  # 假设训练20个epoch，一个epoch就是完整训练一次数据集

for epoch in range(n_epochs):
    model.train()  # 设置为训练模式
    #训练模式会应用一些特定的操作，例如dropout,batch normalization
    for x, y in tr_set: #一个batch循环一次，更新一次参数

        optimizer.zero_grad()  # 清空梯度
        x, y = x.to(device), y.to(device)  # 将数据移到设备上
        pred = model(x)  # 前向传播
        loss = criterion(pred, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
#训练得到的模型参数实际上就是fc1.weight,fc1.bias,fc2.weight,fc2.bias，由model实例自动管理
    # 每个epoch结束后进行验证
    model.eval()  # 设置为评估模式
    total_loss = 0
    with torch.no_grad():  # 关闭梯度计算，评估或推理阶段不需要更改模型参数了
        for x, y in dv_set:

            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.cpu().item() * len(x)  # 一次循环就是计算一次batch的总损失，全部batch循环结束就是计算整个测试集的总损失
    avg_loss = total_loss / len(dv_set.dataset)
    print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {avg_loss:.4f}')

# 测试循环
model.eval()  # 设置为评估模式
preds = []
with torch.no_grad():  # 关闭梯度计算
    for x,y in tt_set: #这里用不到y，但是不加y，x就会变成list
        x = x.to(device)
        pred = model(x)
        preds.append(pred.cpu())  # 将预测结果移回CPU
print(preds)

# 你可以在此处处理或保存preds
preds = torch.cat(preds, dim=0)  # 将所有的预测结果合并成一个大的张量
preds = preds.cpu().numpy().flatten()  # 转换成numpy数组，并展平

# 横轴：时间从第4天开始
time = list(range(4, 4 + len(preds)))  # 从第4天开始，直到最后一天的预测

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(time, preds, label='Predicted', color='blue')  # 绘制预测结果的曲线
# 如果有真实标签y，也可以绘制对比曲线
# plt.plot(time, y, label='True', color='red')  # 如果有真实标签y，也可以绘制对比曲线
plt.xlabel('Time (Days)')
plt.ylabel('Predicted Play Count')
plt.title('Model Fitting')
plt.legend()
plt.show()
