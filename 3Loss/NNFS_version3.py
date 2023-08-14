"""
作者：黄欣
日期：2023年08月07日
"""

# 版本增加了Loss类和CategoricalCrossentropy类（继承了Loss类），只实现了前向传播

import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_input, n_neuron):
        # 用正态分布初始化权重
        self.weight = 0.01 * np.random.randn(n_input, n_neuron)
        # 将bias(偏差)初始化为0
        self.bias = np.zeros(n_neuron)

    def forward(self, input):
        self.output = np.dot(input, self.weight) + self.bias

class Activation_Sigmoid:
    def __init__(self):
        pass

    def forward(self, input):
        # input的大小是nx1，n是Activation输入的sample数量，每个sample只有一个维度。
        # 所以前一个hidden layer必须是Layer_Dense(n, 1)
        self.output = 1 / ( 1 + np.exp(-input) )

class Activation_ReLu:
    def __init__(self):
        pass

    def forward(self,input):
        self.output = np.maximum(0,input)

class Activation_Softmax:
    def __init__(self):
        pass

    def forward(self,input):
        # 要有keepdims=True参数设置
        # 如没有设置，则np.max(input, axis=1)后的列向量会变成行向量，
        # 而行向量长度不与input的每一行长度相同，
        # 则无法广播
        # 进行指数运算之前，从输入值中减去最大值，使输入值更小，从而避免指数运算产生过大的数字
        self.output = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = self.output / np.sum(self.output, axis=1, keepdims=True)

class Loss:
    def __init__(self):
        pass

    # 统一通过调用calculate方法计算损失
    def calculate(self, y_pred, y_ture):
        # 对于不同的损失函数，通过继承Loss父类，并实现不同的forward方法。
        data_loss = np.mean( self.forward(y_pred, y_ture) )
        # 注意，这里计算得到的loss不作为类属性储存，而是直接通过return返回
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        # 多少个样本
        n_sample = len(y_true)

        # 为了防止log(0)，所以以1e-7为左边界
        # 另一个问题是将置信度向1移动，即使是非常小的值，
        # 为了防止偏移，右边界为1 - 1e-7
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        loss = - np.log(y_pred)
        if len(y_true.shape) == 2:# 标签是onehot的编码
            loss = np.sum(loss * y_true,axis=1)
        elif len(y_true.shape) == 1:# 只有一个类别标签
            # 注意loss = loss[:, y_ture]是不一样的，这样会返回一个矩阵
            loss = loss[range(n_sample), y_true]

        # 这里不用求均值，父类中的calculate方法中求均值
        return loss

class Loss_BinaryCrossentropy(Loss):
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        # 多少个样本
        n_sample = len(y_true)
        # 这里要特别注意，书上都没有写明
        # 当只有一对二进制类别时，y_pred大小为(n_sample,1),y_ture大小为(n_sample,)
        # (n_sample,)和(n_sample,1)一样都可以广播，只是(n_sample,)不能转置
        # 所以下面的loss大小会变成(n_sample,n_sample)
        # 当有二对二进制类别时，y_pred大小为(n_sample,2),y_ture大小为(n_sample,2)
        if len(y_true.shape) == 1: # y_true是个行向量
            y_true = y_true.reshape(-1,1)
        # 为了防止log(0)，所以以1e-7为左边界
        # 另一个问题是将置信度向1移动，即使是非常小的值，
        # 为了防止偏移，右边界为1 - 1e-7
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -  np.log(y_pred) * y_true  - np.log(1 - y_pred) * (1 - y_true)
        # 这里的求平均和父类中的calculate求平均的维度不同
        # 这里是对多对的二进制求平均
        # calculate中的求平均是对每个样本可平均
        loss = np.mean(loss, axis=-1)
        return loss

# 生成数据
X, y = spiral_data(samples=100, classes=3)
# 构建一个含三个神经元的Dense层实例
dense1 = Layer_Dense(2,3)
# 构建ReLu激活函数
activation1 = Activation_ReLu()
# 构建一个含4个神经元的Dense层实例
dense2 = Layer_Dense(3,4)
# 构建Softmax激活函数
activation2 = Activation_Softmax()
# 构建损失函数
loss = Loss_CategoricalCrossentropy()

# 前向传播
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dataloss = loss.calculate(activation2.output, y)

# 输出结果
print('loss =',dataloss)

# 计算正确率
soft_output = activation2.output
# 返回最大confidence的类别作为预测类别
prediction = np.argmax(soft_output,axis=1)
# 如果y是onehot编码
if len(y.shape) == 2:
    # 将其变为只有一个标签类别
    y = np.argmax(y,axis=1)

accuracy = np.mean(prediction == y)
print("accurcy =",accuracy)


