"""
作者：黄欣
日期：2023年08月08日
"""

# 版本增加了Dense Layer、Activation Function和Loss的反向传播。

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
        # 因为要增加backward方法，
        # Layer_Dense的输出对输入（input）的偏导是self.weight，
        # 面Layer_Dense的输出对self.weight的偏导是输入（input）
        # 所以要在forward中增加self.input属性
        self.input = input
        self.output = np.dot(input, self.weight) + self.bias

    def backward(self, dvalue):
        # dvalue是loss对下一层（Activation）的输入（input）的导数，
        # 也就是loss对这一层（Layer_Dense）的输出（output）的导数，
        # 这里会用到链式法则

        # 在本层中，希望求得的是loss对这一层（Layer_Dense）的self.weight的导数
        # 这便找到了self.weight优化的方向（negative gradient direction）

        # 这里要考虑到self.dweight的大小要与self.weight一致，因为方便w - lr * dw公式进行优化
        # 假设input只有一个sample，大小为1xa，weight大小为axb，则output大小为1xb，
        # 因为loss是标量，所以dvalue = dloss/doutput大小即为output的大小(1xb)，
        # 所以dweight的大小为(1xa).T * (1xb) = axb,大小和weight一致。
        # 注意：当input有多个sample时（一个矩阵输入），则dweight为多个axb矩阵相加。
        self.dweight = np.dot(self.input.T, dvalue)

        # 在本层中，希望求得的是loss对这一层（Layer_Dense）的self.input的导数
        # 以便作为下一层的backward方法中的dvalue参数，

        # 因为loss是标量，所以dinput大小即为intput的大小(1xa)，
        # dvalue = dloss/doutput大小即为output的大小(1xb)，
        # weight大小为axb
        # 所以1xa = (1xb) * (axb).T
        self.dinput = np.dot(dvalue, self.weight.T)

        # 像self.dinput一样，self.dbias可以通过矩阵乘法实现，
        # self.dbias = np.dot( dvalue, np.ones( ( len(self.bias), len(self.bias) ) ) )
        # 但有更快更简单的实现
        self.dbias = np.sum(dvalue, axis=0, keepdims=True)# 此处不要keepdims=True也行，因为按0维相加还是行向量

class Activation_Sigmoid:
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input

        # input的大小是nx1，n是Activation输入的sample数量，每个sample只有一个维度。
        # 所以前一个hidden layer必须是Layer_Dense(n, 1)
        self.output = 1 / ( 1 + np.exp(-input) )

    def backward(self, dvalue):
        # 这里也可以用矩阵计算，但dinput、dvalue、output大小相同，
        # 可以直接按元素对应相乘。
        self.dinput = dvalue * self.output * ( 1 - self.output )

class Activation_ReLu:
    def __init__(self):
        pass

    def forward(self,input):
        self.input = input
        self.output = np.maximum(0,input)

    def backward(self, dvalue):
        # self.input和self.output形状是一样的
        # 那么dinput大小=doutput大小=dvalue大小
        # 可以用mask来更快实现，而不用矩阵运算
        self.dinput = dvalue.copy()
        self.dinput[self.input < 0] = 0

class Activation_Softmax:
    def __init__(self):
        pass

    def forward(self,input):
        self.input = input

        # 要有keepdims=True参数设置
        # 如没有设置，则np.max(input, axis=1)后的列向量会变成行向量，
        # 而行向量长度不与input的每一行长度相同，
        # 则无法广播
        # 进行指数运算之前，从输入值中减去最大值，使输入值更小，从而避免指数运算产生过大的数字
        self.output = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = self.output / np.sum(self.output, axis=1, keepdims=True)

    def backward(self, dvalue):
        # input和output大小相同都为1xa，
        # loss是标量，那么dinput和doutput（即dvalue）大小相同都为1xa，
        # output对input的导数为一个axa的方阵

        # 相同大小的空矩阵
        self.dinput = np.empty_like(dvalue)
        # 对每个samlpe（每一行）循环
        for each, (single_output, single_dvalue) in enumerate(zip(self.output, dvalue)):
            # 这里是(1xa) * (axa) = 1xa是行向量
            # 这里要先将1xa向量变为1xa矩阵
            # 因为向量没有转置（.T操作后还是与原来相同），
            # np.dot接收到向量后，会调整向量的方向，但得到的还是向量（行向量）,就算得到列向量也会表示成行向量
            # np.dot接收到1xa矩阵后，要考虑前后矩阵大小的匹配，不然要报错,最后得到的还是矩阵
            single_output = single_output.reshape(1, -1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output.T,single_output)
            # 因为single_dvalue是行向量，dot运算会调整向量的方向
            # 所以np.dot(single_dvalue, jacobian_matrix)和np.dot(jacobian_matrix， single_dvalue)
            # 得到的都是一个行向量，但两都的计算方法不同，得到的值也不同
            # np.dot(jacobian_matrix, single_dvalue)也是对的，这样得到的才是行向量，
            # 而不是经过dot将列向量转置成行向量
            self.dinput[each] = np.dot(jacobian_matrix, single_dvalue)


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

        # loss是一个列向量，每一行是一个样本,
        # 这里不用求均值，父类中的calculate方法中求均值
        return loss

    def backward(self, y_pred, y_true):
        n_sample = len(y_true)
        if len(y_true.shape) == 2:  # 标签是onehot的编码
            label = y_true
        elif len(y_true.shape) == 1:  # 只有一个类别标签
            # 将标签改成onehot的编码
            label = np.zeros((n_sample, len(y_pred[0])))
            label[range(n_sample), y_true] = 1
        self.dinput = - label / y_pred
        # 每个样本除以n_sample，因为在优化的过程中要对样本求和
        self.dinput = self.dinput / n_sample


class Loss_BinaryCrossentropy(Loss):
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
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

    def backward(self, y_pred, y_true):
        # 样本个数
        n_sample = len(y_true)
        # 二进制输出个数
        n_output = len(y_pred[0])
        # 这里要特别注意，书上都没有写明
        # 当只有一对二进制类别时，y_pred大小为(n_sample,1),y_ture大小为(n_sample,)
        # (n_sample,)和(n_sample,1)一样都可以广播，只是(n_sample,)不能转置
        # 所以下面的loss大小会变成(n_sample,n_sample)
        # 当有二对二进制类别时，y_pred大小为(n_sample,2),y_ture大小为(n_sample,2)
        if len(y_true.shape) == 1:  # y_true是个行向量
            y_true = y_true.reshape(-1, 1)
        # 注意：BinaryCrossentropy之前都是Sigmoid函数
        # Sigmoid函数很容易出现0和1的输出
        # 所以以1e-7为左边界
        # 另一个问题是将置信度向1移动，即使是非常小的值，
        # 为了防止偏移，右边界为1 - 1e-7
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # 千万不要与成下面这样，因为-y_true优先级最高，而y_true是uint8，-1=>255
        # 这个bug我找了很久，要重视
        # self.dinput = -y_true / y_pred_clip + (1 - y_true) / (1 - y_pred_clip)) / n_output
        self.dinput = -(y_true / y_pred_clip - (1 - y_true) / (1 - y_pred_clip)) / n_output
        # 每个样本除以n_sample，因为在优化的过程中要对样本求和
        self.dinput = self.dinput / n_sample

########################################
# 生成数据
X, y = spiral_data(samples=100, classes=2)
#########################################

########################################################
# 构建一个含三个神经元的Dense层实例
dense1 = Layer_Dense(2,4)
# 构建ReLu激活函数
activation1 = Activation_ReLu()
# 构建一个含4个神经元的Dense层实例
dense2 = Layer_Dense(4,1)
# 构建Softmax激活函数
activation2 = Activation_Sigmoid()
# 构建损失函数
loss = Loss_BinaryCrossentropy()

# 前向传播
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dataloss = loss.calculate(activation2.output, y)

# 反向传播
loss.backward(activation2.output, y)
activation2.backward(loss.dinput)
dense2.backward(activation2.dinput)
print(dense2.dinput[40:50])
#print(activation2.dinput[40:50])

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
#################################################

#############################################3
# 构建一个含三个神经元的Dense层实例
dense1 = Layer_Dense(2,4)
# 构建ReLu激活函数
activation1 = Activation_ReLu()
# 构建一个含4个神经元的Dense层实例
dense2 = Layer_Dense(4,2)
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

# 反向传播
loss.backward(activation2.output, y)
activation2.backward(loss.dinput)
dense2.backward(activation2.dinput)
print(dense2.dinput[40:50])
#print(activation2.dinput[40:50])

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
########################################################







