"""
作者：黄欣
日期：2023年08月06日
"""
import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

# 生成数据
X, y = spiral_data(samples=100, classes=3)
# 查看数据大小
print(X.shape,y.shape)
# # 设置了图形的参数，以y数组中的值作颜色，并使用brg三颜鎟
# # 并注意，Matplotlib内置的颜色映射名称为'brg',并不是常用的'rgb'顺序
# plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
# # 显示图形
# plt.show()

class Layer_Dense:
    def __init__(self, n_input, n_neuron):
        # 用正态分布初始化权重
        self.weight = 0.01 * np.random.randn(n_input, n_neuron)
        # 将bias(偏差)初始化为0
        self.bias = np.zeros(n_neuron)

    def forward(self, input):
        self.output = np.dot(input, self.weight) + self.bias

# 构建一个含三个神经元的Dense层实例
dense = Layer_Dense(2,3)
# 前向传播
dense.forward(X);
# 输出结果
print(dense.output[:5])