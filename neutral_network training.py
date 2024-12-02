import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd

# 自定义数据集类 (假设你有一个数据文件和读取方式)
class MyDataset(Dataset):
    def __init__(self, file):
        # 假设file是数据文件路径，需要根据实际情况加载数据
        self.data = self.load_data(file)

    def load_data(self, file):
        # 实现数据加载逻辑（例如，读取CSV文件等）
        data = pd.read_csv(file)  # 这里的 file 就是传入的文件路径
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 这里假设数据集按日期升序排列，idx 对应的是当前时间点
        # 假设过去 10 天作为输入，预测第 11 天
        if idx >= 3:  # 确保数据足够长
            x = self.data.iloc[idx - 3:idx, 1].values  # 获取过去 10 天的播放量
            y = self.data.iloc[idx, 1]  # 获取当前日期的播放量
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return None




# 定义一个简单的神经网络模型 (假设是一个简单的全连接网络)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 假设输入特征是10维
        self.fc2 = nn.Linear(64, 1)   # 输出1维 (回归问题)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
file = "youtube_play_counts.csv"  # 数据文件路径
dataset = MyDataset(file)
tr_set = DataLoader(dataset, batch_size=16, shuffle=True)  # 训练集
dv_set = DataLoader(dataset, batch_size=16, shuffle=False) # 验证集
tt_set = DataLoader(dataset, batch_size=16, shuffle=False) # 测试集

# 模型构建
model = MyModel().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练和验证循环
n_epochs = 20  # 假设训练20个epoch

for epoch in range(n_epochs):
    model.train()  # 设置为训练模式
    for x, y in tr_set:
        optimizer.zero_grad()  # 清空梯度
        x, y = x.to(device), y.to(device)  # 将数据移到设备上
        pred = model(x)  # 前向传播
        loss = criterion(pred, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    # 每个epoch结束后进行验证
    model.eval()  # 设置为评估模式
    total_loss = 0
    with torch.no_grad():  # 关闭梯度计算
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.cpu().item() * len(x)  # 累加损失
    avg_loss = total_loss / len(dv_set.dataset)
    print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {avg_loss:.4f}')

# 测试循环
model.eval()  # 设置为评估模式
preds = []
with torch.no_grad():  # 关闭梯度计算
    for x in tt_set:
        x = x.to(device)
        pred = model(x)
        preds.append(pred.cpu())  # 将预测结果移回CPU

# 你可以在此处处理或保存preds
