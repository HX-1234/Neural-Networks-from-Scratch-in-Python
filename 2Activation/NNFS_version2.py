"""
作者：黄欣
日期：2023年08月07日
"""

# 版本增加了Activation类，只实现了前向传播

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

class Activation_Softmax:
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

# 生成数据
X, y = spiral_data(samples=100, classes=3)
# 构建一个含三个神经元的Dense层实例
dense = Layer_Dense(2,3)
# 构建Softmax激活函数
activation1 = Activation_Softmax()

# 前向传播
dense.forward(X)
activation1.forward(dense.output)
# 输出结果
print(activation1.output[:5])


