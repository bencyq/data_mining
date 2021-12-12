import numpy as np
import pandas as pd
import torch
from torch import nn


def linear_regression(x, y, D_inputs, D_outputs, epoch=5000, lr=0.001):
    # 构建模型
    model = nn.Sequential(nn.Linear(D_inputs, D_outputs))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 训练模型
    for t in range(epoch):
        # train_set
        model.zero_grad()  # 梯度清零
        y_pred = model(x[:-10, :])
        loss = loss_fn(y_pred, y[:-10, :])
        loss.backward()

        # test_set
        y_test_pred = model(x[-10:, :])
        test_loss = loss_fn(y_test_pred, y[-10:, :])
        # 参数更新
        with torch.no_grad():  # 参数更新方式
            for parm in model.parameters():
                parm -= lr * parm.grad
        if t % 100 == 0:
            print(f'iter: {t}\ttrain_loss: {loss}\ttest_loss: {test_loss}')

    torch.save(model.state_dict(), 'train1.pt')
    print(model.state_dict())
    return


train_df = pd.read_excel('train1-result.xlsx', header=None, index_col=None)
train_x1 = np.array((train_df))
train_x1 = train_x1.astype(float)
train_x1 = torch.from_numpy(train_x1)
train_x1 = torch.tensor(train_x1, dtype=torch.float32)
train_df = pd.read_excel('train2-result.xlsx', header=None, index_col=None)
train_x2 = np.array((train_df))
train_x2 = train_x2.astype(float)
train_x2 = torch.from_numpy(train_x2)
train_x2 = torch.tensor(train_x2, dtype=torch.float32)
train_y = np.array(pd.read_excel('preprocessed.xlsx', header=None, index_col=None))
train_y1 = torch.tensor(torch.from_numpy(train_y[2:, -2].astype(float)), dtype=torch.float32).resize(138, 1) / 10
train_y2 = torch.tensor(torch.from_numpy(train_y[2:, -1].astype(float)), dtype=torch.float32).resize(138, 1) / 10

linear_regression(train_x1, train_y1, 35, 1)
