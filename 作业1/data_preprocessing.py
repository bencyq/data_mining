import pandas as pd
import numpy as np


def data_cleaning(df):  # 数据清洗，用众数填充所有的"未答"选项
    column = df.shape[1]
    for col in range(column):
        if '未答' in list(df[:][col]):
            df[:][col].replace('未答', df[:][col].value_counts().index[0], inplace=True)
    return df


def data_integration(df1, df2):  # 数据集成，将多个数据源整合到一起，并将属性不同但是意义一样的部分合并（如线代和线性代数）
    df = pd.concat([df1, df2[2:][:]], ignore_index=True)
    # 把第一列男女转换为1 0
    for i in range(2, df.shape[0]):
        if df[0][i] == '女':
            df[0][i] = 0
        elif df[0][i] == '男':
            df[0][i] = 1
        else:
            print(i, df[i][10])
    # 把第十一列是否转换为1 0
    for i in range(2, df.shape[0]):
        if df[10][i] == '否':
            df[10][i] = 0
        elif df[10][i] == '是':
            df[10][i] = 1
        else:
            print(i, df[i][10])

    # 把第十二列的高数、线代转换为0 1
    for i in range(2, df.shape[0]):
        if '高' in df[11][i]:
            df[11][i] = 0
        else:
            df[11][i] = 1
    arr = np.array(df)
    lis = list()
    for i in range(arr.shape[1]):
        if not str(arr[5, i])[-1].isdigit():
            lis.append(i)
    arr = np.delete(arr, lis, axis=1)
    arr = arr[3:].astype(float)
    arr = arr.T
    corrcoef = np.corrcoef(arr)
    for i in range(corrcoef.shape[0]):
        for j in range(i, corrcoef.shape[1]):
            if 0.99 > corrcoef[i][j] > 0.80:
                print(i, j, corrcoef[i][j])
    # pd.DataFrame(corrcoef).to_excel('corrcoef.xlsx')
    return df


# def data_reduction():


def data_preprocessing():
    df = pd.read_excel('v5-问题回答和成绩-C语言_周二.xlsx', index_col=None, header=None)
    df2 = pd.read_excel('v5-问题回答和成绩-C语言_周五.xlsx', index_col=None, header=None)
    df_new = data_integration(data_cleaning(df), data_cleaning(df2))
    df_new.to_excel('preprocessed.xlsx', header=None, index=None)


if __name__ == '__main__':
    data_preprocessing()
