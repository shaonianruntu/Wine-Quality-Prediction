import os
import sys
import numpy as np 
import pandas as pd 

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import matplotlib as mpl 
import matplotlib.pyplot as plt 


#对数据集中的样本属性进行分割，制作X和Y矩阵 
def feature_label_split(pd_data):
    #行数、列数
    row_cnt, column_cnt = pd_data.shape
    #生成新的X、Y矩阵
    X = np.empty([row_cnt, column_cnt-1])       #生成两个随机未初始化的矩阵
    Y = np.empty([row_cnt, 1])
    for i in range(0, row_cnt):
        row_array = redwine_data.iloc[i, ]
        X[i] = np.array(row_array[0:-1])
        Y[i] = np.array(row_array[-1])
    return X, Y


#把特征数据进行标准化为均匀分布
def uniform_norm(X_in):
    X_max = X_in.max(axis=0)
    X_min = X_in.min(axis=0)
    X = (X_in-X_min)/(X_max-X_min)
    return X


#主函数
if __name__ == "__main__":
    
    #读取样本数据
    redwine_data = pd.read_csv("winequality-red.csv", sep=";")
    #样本数据进行X、Y矩阵分离
    X, Y = feature_label_split(redwine_data)
    #对X矩阵进行归一化
    unif_X = uniform_norm(X)
    #对样本数据进行训练集和测试集的划分
    unif_trainX, unif_testX, train_Y, test_Y = train_test_split(unif_X, Y, test_size=0.3, random_state=0)
    
    
    #模型训练
    model = Ridge()     #L2正则的线性回归
    model.fit(unif_trainX, train_Y)
    
    #模型评估
    print("训练集上效果评估 >>")
    r2 = model.score(unif_trainX, train_Y)
    print("R^2系数 ", r2)
    train_pred = model.predict(unif_trainX)
    mse = mean_squared_error(train_Y, train_pred) 
    print("均方误差 ", mse)

    print("\n测试集上效果评估 >>")
    r2 = model.score(unif_testX, test_Y)
    print("R^2系数 ", r2)
    test_pred = model.predict(unif_testX)
    mse = mean_squared_error(test_Y, test_pred)        
    #等价于 mse = sum((test_pred-test_Y)**2) / test_Y.shape[0]
    print("均方误差", mse)     