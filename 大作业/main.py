import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# datasets path and df names
path = [
    'dataset\\demographic.csv',
    'dataset\\examination.csv',
    'dataset\\questionnaire.csv',
    'dataset\\labs.csv',
    'dataset\\diet.csv'
]

dfname = [
    'dm',
    'exam',
    'qs',
    'lab',
    'diet'
]
from imblearn.over_sampling import SMOTE

# 读入数据

df = {}
dfn = dict(zip(dfname, path))
df = {key: pd.read_csv(value) for key, value in dfn.items()}

Xs = {k: v for k, v in df.items() if k in ['dm', 'exam', 'labs']}

dfs = Xs.values()

from functools import partial, reduce

inner_merge = partial(pd.merge, how='inner', on='SEQN')

c = reduce(inner_merge, dfs)
# 查找是否有重复的序列号
print(c.SEQN.duplicated().value_counts())

# 展示数据
qs = df['qs'][['SEQN', 'MCQ160F']]
print(qs)

# 拼接矩阵
c = pd.merge(c, qs, how='left', on='SEQN')
print(c)

print(c.MCQ160F.value_counts())

"""
Exclude rows with null values or NA for MCQ160F
The prediction target in the dataset is MCQ160F, a questionnaire question "Has a doctor or other health professional ever told you that you had a stroke?"
"""
# MCQ160F (target feature): exclude null values and NA
c = c[(c.MCQ160F.notnull()) & (c.MCQ160F != 9)]

# check MCQ160F
c.MCQ160F.describe()