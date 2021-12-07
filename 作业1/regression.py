import numpy as np
import pandas as pd
import torch
from torch import nn


def linear_regression(x, y, test_x, test_y, D_inputs=49, D_outputs=7, epoch=10001, temp='none', lr=0.0001):
    # 构建模型
    model = nn.Sequential(nn.Linear(D_inputs, D_outputs))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print(temp)
    # 训练模型
    for t in range(epoch):
        # train_set
        model.zero_grad()  # 梯度清零
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        # test_set
        y_pred_test = model(test_x)
        test_loss = loss_fn(y_pred_test, test_y)

        # 参数更新
        with torch.no_grad():  # 参数更新方式
            for parm in model.parameters():
                parm -= lr * parm.grad
        if t % 100 == 0:
            print('iter: {}\ttrain_loss: {}\ttest_loss: {}'.format(t, loss, test_loss))

    out = model(x[666, :])
    out_pred = model(test_x[-1, :])
    print(out)
    print(y[666])
    print('pred:{}'.format(out_pred))
    torch.save(model.state_dict(), '{}.pt'.format(temp))
    return


file = 'train1'
train_df = pd.read_excel(file + 'xlsx', header=None, index_col=None)
train_x = np.array((train_df))
train_x = train_x.astype(float)
train_x = torch.from_numpy(train_x)
train_x = torch.tensor(train_x, dtype=torch.float32)

