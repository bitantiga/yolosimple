import torch
import torch.nn as nn
import torch.optim as optim
# 定义一个张量
w = torch.tensor([[1.5, 2.1], [3.0, 4.5]], requires_grad=True)
x=torch.tensor([[1,1],[1,1]])
# 对张量进行截断操作

y = nn.functional.relu((w+x))#.clamp(2.0, 4.0)

# 计算梯度
y.sum().backward()

# 输出张量和梯度
print("x:", x)
print("y:", y)
print("grad:", w.grad)
# 定义 MLP 的模型结构
class MLP(nn.Module):
    def __init__(self, input_dim, n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden)
        #self.conv=nn.Conv1d(2,2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #out=self.conv()
        out = self.fc1(x)

        out = self.relu(out).clamp(0.0,0.6)
        print(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 数据集
X = torch.tensor([[0, 0,1], [0,3, 1], [1,2, 0], [1,0, 1],[1,3,3], [1,3, 1], [0,3, 2], [1,3, 2], [0,0, 2]], dtype=torch.float)
y = torch.tensor([0, 1, 1, 0,1,1,0,0,0], dtype=torch.float).reshape([-1,1])
x1=torch.tensor([[1,3,3], [1,3, 1], [0,3, 1], [1,0, 1]], dtype=torch.float)
# 定义超参数
lr = 0.1
n_epochs = 0
input_dim = 3
n_hidden = 4

# 初始化模型
model = MLP(input_dim, n_hidden)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 训练模型
for epoch in range(n_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失函数值
    loss = criterion(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    print( model.fc1.weight.grad)
    loss.backward()
    print( model.fc1.weight.grad)
    # 更新权重
    optimizer.step()
    print( model.fc1.weight.grad)
    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))

# 模型评估
with torch.no_grad():
    y_pred = model(X)

    y_pred_cls = torch.round(y_pred)
    acc = (y_pred_cls == y).sum().item() / len(y)
    
    print('Accuracy: {:.2f}%'.format(acc * 100))
