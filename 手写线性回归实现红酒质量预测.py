import sys
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


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


#线性回归模型
class linear_regression():

    def fit(self, train_X_in, train_Y, learning_rate=0.03, lamda=0.03, regularization="l2"):
        #样本个数、样本的属性个数
        case_cnt, feature_cnt = train_X_in.shape 
        #X矩阵添加X0向量
        train_X = np.c_[train_X_in, np.ones(case_cnt, )]
        #初始化待调参数theta
        self.theta = np.zeros([feature_cnt+1, 1])

        max_iter_num = sys.maxsize      #最多迭代次数 
        step = 0                        #当前已经迭代的次数
        pre_step = 0                    #上一次得到较好学习误差的迭代学习次数

        last_error_J = sys.maxsize      #上一次得到较好学习误差的误差函数值
        threshold_value = 1e-6          #定义在得到较好学习误差之后截止学习的阈值
        stay_threshold_times = 10       #定义在得到较好学习误差之后截止学习之前的学习次数

        for step in range(0, max_iter_num):
            #预测值
            pred = train_X.dot(self.theta)
            #损失函数
            J_theta = sum((pred-train_Y)**2) / (2*case_cnt)
            #更新参数theta
            self.theta -= learning_rate*(lamda*self.theta + (train_X.T.dot(pred-train_Y))/case_cnt)          

            #检测损失函数的变化值，提前结束迭代
            if J_theta < last_error_J - threshold_value:
                last_error_J = J_theta
                pre_step = step
            elif step - pre_step > stay_threshold_times:
                break

            #定期打印，方便用户观察变化 
            if step % 100 == 0:
                print("step %s: %.6f" % (step, J_theta))


    def predict(self, X_in):
        case_cnt = X_in.shape[0]
        X = np.c_[X_in, np.ones(case_cnt, )]
        pred = X.dot(self.theta)
        return pred


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
    model = linear_regression()
    model.fit(unif_trainX, train_Y, learning_rate=0.1)
    #模型预测
    test_pred = model.predict(unif_testX)
    test_pred_error = sum((test_pred-test_Y)**2) / (2*unif_testX.shape[0])
    print("Test error is %.6f" % (test_pred_error))




