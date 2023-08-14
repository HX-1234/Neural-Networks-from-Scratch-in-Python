"""
作者：黄欣
日期：2023年08月13日
"""

# 本版本将加入模型的测试方法，包括分批处理（batch）




import numpy as np
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt
import nnfs


class Layer_Input:
    def __init__(self):
        pass

    def forward(self, input):
        self.output = input

class Layer_Dense:
    def __init__(self, n_input, n_neuron, weight_L1=0., weight_L2=0., bias_L1=0., bias_L2=0.):
        # 用正态分布初始化权重
        self.weight = 0.1 * np.random.randn(n_input, n_neuron)
        # 将bias(偏差)初始化为0
        # self.bias = np.zeros(n_neuron)
        self.bias = np.zeros((1, n_neuron))
        self.weight_L1 = weight_L1
        self.weight_L2 = weight_L2
        self.bias_L1 = bias_L1
        self.bias_L2 = bias_L2

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
        self.dbias = np.sum(dvalue, axis=0, keepdims=True)  # 此处不要keepdims=True也行，因为按0维相加还是行向量

        # 正则项的梯度
        if self.weight_L2 > 0:
            self.dweight += 2 * self.weight_L2 * self.weight
        if self.bias_L2 > 0:
            self.dbias += 2 * self.bias_L2 * self.bias
        if self.weight_L1 > 0:
            dL = np.ones_like(self.weight)
            dL[self.weight < 0] = -1
            self.dweight += self.weight_L1 * dL
        if self.bias_L1 > 0:
            dL = np.ones_like(self.bias)
            dL[self.bias < 0] = -1
            self.dbias += self.bias_L1 * dL

class Activation_Sigmoid:
    def __init__(self):
        pass

    # prediction方法输出预测类别（分类）
    def prediction(self, output):
        # output > 0.5反回的是二进制值
        # 乘1变成数值
        return ( output > 0.5 ) * 1

    def forward(self, input):
        self.input = input

        # input的大小是nx1，n是Activation输入的sample数量，每个sample只有一个维度。
        # 所以前一个hidden layer必须是Layer_Dense(n, 1)
        self.output = 1 / (1 + np.exp(- (self.input)))

    def backward(self, dvalue):
        # 这里也可以用矩阵计算，但dinput、dvalue、output大小相同，
        # 可以直接按元素对应相乘。
        self.dinput = dvalue * self.output * (1 - self.output)


class Activation_ReLu:
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)

    def backward(self, dvalue):
        # self.input和self.output形状是一样的
        # 那么dinput大小=doutput大小=dvalue大小
        # 可以用mask来更快实现，而不用矩阵运算
        self.dinput = dvalue.copy()
        self.dinput[self.input < 0] = 0


class Activation_Softmax:
    def __init__(self):
        pass

    # prediction方法输出预测类别（分类）
    def prediction(self, output):
        return np.argmax(output, axis=1, keepdims=True)

    def forward(self, input):
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
            # 显然这两种计算法算到的dinput大小是一样的
            # 这里是(1xa) * (axa) = 1xa是行向量
            # 这里要先将1xa向量变为1xa矩阵
            # 因为向量没有转置（.T操作后还是与原来相同），
            # np.dot接收到向量后，会调整向量的方向，但得到的还是向量（行向量）,就算得到列向量也会表示成行向量
            # np.dot接收到1xa矩阵后，要考虑前后矩阵大小的匹配，不然要报错,最后得到的还是矩阵
            single_output = single_output.reshape(1, -1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output.T, single_output)
            # 因为single_dvalue是行向量，dot运算会调整向量的方向
            # 所以np.dot(single_dvalue, jacobian_matrix)和np.dot(jacobian_matrix， single_dvalue)
            # 得到的都是一个行向量，但两都的计算方法不同，得到的值也不同
            # np.dot(jacobian_matrix, single_dvalue)也是对的，这样得到的才是行向量，
            # 而不是经过dot将列向量转置成行向量
            self.dinput[each] = np.dot(jacobian_matrix, single_dvalue)

class Activation_Linear:
    def __init__(self):
        pass

    # prediction方法输出值（回归）
    def prediction(self, output):
        return output

    def forward(self, input):
        self.input = input
        self.output = self.input

    def backward(self, dvalue):
        # 注意与self.dinput = dvalue（目前还未发现这样无不可）
        # 这意味着 dinput 和 dvalue 指向同一个对象，因此对 dinput 的任何更改都会影响原始的 dvalue 对象
        # 而对dvalue进行运算如乘1，则和下面代码一样
        self.dinput = dvalue.copy()


class Loss:
    def __init__(self):
        pass

    # 在求loss时需要知参别些层里面有可以训练参数，可以正则化
    def save_trainable_layer(self, trainable_layer):
        self.trainable_layer = trainable_layer

    # 统一通过调用calculate方法计算损失
    def calculate(self, y_pred, y_ture, *, add_regular_loss=False):
        # 对于不同的损失函数，通过继承Loss父类，并实现不同的forward方法。
        sample_loss = self.forward(y_pred, y_ture)
        data_loss = np.mean(sample_loss)
        # 加入了batch，所以要计算累计的损失和已训练过的样本数
        self.cumulate_dataloss += np.sum(sample_loss)
        self.cumulate_num += len(sample_loss)

        # 在加入正则代码后，可以求得正则损失
        # 注意之前版本调用regularization_loss(layer)
        # 但这个版本有了self.trainable_layer，可直接找到Dense层（有参数）
        regularization_loss = self.regularization_loss()
        if not add_regular_loss:
            # 在测试模型性能时只关心data_loss
            regularization_loss = 0
        # 注意，这里计算得到的loss不作为类属性储存，而是直接通过return返回
        return data_loss, regularization_loss

    def calculate_cumulate(self, *, add_regularization=False):

        # 返回平均损失
        data_loss = self.cumulate_dataloss / self.cumulate_num

        regularization_loss = 0
        if add_regularization:
            regularization_loss = self.regularization_loss()
        return data_loss, regularization_loss

    def clean_cumulate(self):
        self.cumulate_dataloss = 0
        self.cumulate_num = 0

    def regularization_loss(self):
        # 默认为0
        regularization_loss = 0
        for layer in self.trainable_layer:
            # 如果存在L1的loss
            if layer.weight_L1 > 0:
                regularization_loss += layer.weight_L1 * np.sum(np.abs(layer.weight))
            if layer.bias_L1 > 0:
                regularization_loss += layer.bias_L1 * np.sum(np.abs(layer.bias))
            # 如果存在L2的loss
            if layer.weight_L2 > 0:
                regularization_loss += layer.weight_L2 * np.sum(layer.weight ** 2)
            if layer.bias_L2 > 0:
                regularization_loss += layer.bias_L2 * np.sum(layer.bias ** 2)

        return regularization_loss

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
        if len(y_true.shape) == 2:  # 标签是onehot的编码
            loss = np.sum(loss * y_true, axis=1)
        elif len(y_true.shape) == 1:  # 只有一个类别标签
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
        if len(y_true.shape) == 1:  # y_true是个行向量
            y_true = y_true.reshape(-1, 1)
        # 为了防止log(0)，所以以1e-7为左边界
        # 另一个问题是将置信度向1移动，即使是非常小的值，
        # 为了防止偏移，右边界为1 - 1e-7
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = - ( np.log(y_pred_clip) * y_true + np.log(1 - y_pred_clip) * (1 - y_true) )
        # 这里的求平均和父类中的calculate求平均的维度不同
        # 这里是对多对的二进制求平均
        # calculate中的求平均是对每个样本可平均
        loss = np.mean(loss, axis=-1)
        return loss

    def backward(self, y_pred, y_true):
        # 样本个数
        n_sample = len(y_pred)
        # 二进制输出个数
        n_output = len(y_pred[0])
        # 这里要特别注意，书上都没有写明
        # 当只有一对二进制类别时，y_pred大小为(n_sample,1),y_ture大小为(n_sample,)
        # (n_sample,)和(n_sample,1)一样都可以广播，只是(n_sample,)不能转置
        # 所以下面的loss大小会变成(n_sample,n_sample)
        # 当有二对二进制类别时，y_pred大小为(n_sample,2),y_ture大小为(n_sample,2)
        if len(y_true.shape) == 1:  # y_true是个行向量
            y_true = y_true.reshape(-1, 1)
        if len(y_true.shape) == 2 and y_true.shape[0] == 1:
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
        self.dinput = -( y_true / y_pred_clip - (1 - y_true) / (1 - y_pred_clip)) / n_output
        # 每个样本除以n_sample，因为在优化的过程中要对样本求和
        self.dinput = self.dinput / n_sample

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # 注意：Activation_Softmax_Loss_CategoricalCrossentropy类中是调用forward计算loss
    # 因为它没有继承Loss类
    def forward(self, input, y_true):
        self.activation.forward(input)
        # 该类的output属性应该是Activation_Softmax()的输出
        self.output = self.activation.output
        # 该类返回的是loss
        return self.loss.calculate(self.output, y_true)

    # 其实y_pred一定等于self.output，但为了与之前代码一致
    def backward(self, y_pred, y_true):
        # 样本个数
        n_sample = len(y_true)
        if len(y_true.shape) == 2:  # onehot编码
            # 直接套公式
            self.dinput = y_pred - y_true
        elif len(y_true.shape) == 1:  # 只有一个类别
            self.dinput = y_pred.copy()
            # 需将每一行中y_true类别（索引）中的-1，其它-0（不操作）
            self.dinput[range(n_sample), y_true] -= 1
        # 每个样本除以n_sample，因为在优化的过程中要对样本求和
        self.dinput = self.dinput / n_sample


class Activation_Sigmoid_Loss_BinaryCrossentropy():
    def __init__(self):
        self.activation = Activation_Sigmoid()
        self.loss = Loss_BinaryCrossentropy()

    def forward(self, input, y_true):
        self.activation.forward(input)
        # 类的output是Sigmoid的输出
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, y_pred, y_true):
        # 样本数量
        n_sample = len(y_pred)
        # 这里要特别注意，书上都没有写明
        # 当只有一对二进制类别时，y_pred大小为(n_sample,1),y_ture大小为(n_sample,)
        # (n_sample,)和(n_sample,1)一样都可以广播，只是(n_sample,)不能转置
        # 所以下面的loss大小会变成(n_sample,n_sample)
        # 当有二对二进制类别时，y_pred大小为(n_sample,2),y_ture大小为(n_sample,2)
        if len(y_true.shape) == 1:  # y_true是个行向量
            y_true = y_true.reshape(-1, 1)
        # 二进制输出个数
        J = len(y_pred[0])
        # y_true中每一行都有J个1或0的二进制值，1代表正例，0代表负例。
        self.dinput = (y_pred - y_true) / J

        # 优化时要将所有样本相加，为了梯度与样本数量无关，这里除以样本数
        self.dinput /= n_sample

class Loss_MeanSquaredError(Loss):
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        # 输出变量的维度
        loss = np.mean( (y_pred - y_true) ** 2, axis=-1 )
        return loss

    def backward(self, y_pred, y_true):
        # 样本个数
        n_sample = len(y_pred)
        # 输出维度
        n_output = len(y_true[0])
        self.dinput = 2 / n_output * (y_pred - y_true)
        # 这里要非常注意，之前的解释都错了
        # 在loss类的calculate方法中有data_loss = np.mean( self.forward(prediction, y) )
        # 有一个对样本求均值的过程，即有一个除以样本个数的计算，所以求导后，除以样本个数来传递过来。
        self.dinput /= n_sample

class Optimizer_SGD():
    # 初始化方法将接收超参数，从学习率开始，将它们存储在类的属性中
    def __init__(self, learning_rate = 1.0, decay = 0., momentum=0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iteration = 0
        self.momentum = momentum

    def pre_update_param(self):
        # 这种衰减的工作原理是取步数和衰减比率并将它们相乘。
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1 / (1 + self.decay * self.iteration))

    # 给一个层对象参数，执行最基本的优化
    def update_param(self, layer):

        deta_weight = layer.dweight
        deta_bias = layer.dbias

        # 如果使用momentum
        if self.momentum:
            # 如果还没有累积动量
            if not hasattr(layer, "dweight_cumulate"):
                # 注意：这里是往layer层里加属性
                # 这很容易理解，历史信息肯定是要存在对应的对像中
                layer.dweight_cumulate = np.zeros_like(layer.weight)
                layer.dbias_cumulate = np.zeros_like(layer.bias)
            deta_weight += self.momentum * layer.dweight_cumulate
            layer.dweight_cumulate = deta_weight
            deta_bias += self.momentum * layer.dbias_cumulate
            layer.dbias_cumulate = deta_bias
        layer.weight -= self.current_learning_rate * deta_weight
        # (64,) = (64,) + (1,64) >> (1,64)
        # (64,) += (1,64) >> 无法广播
        # (1, 64) = (64,) + (1,64) >> (1,64)
        # (1, 64) += (64,) >> (1,64)
        # 所以修改了dense中
        # self.bias = np.zeros(n_neuron) => self.bias = np.zeros((1, n_neuron))
        layer.bias -= self.current_learning_rate * deta_bias

    def post_update_param(self):
        self.iteration += 1

class Optimizer_Adagrad():
    # 初始化方法将接收超参数，从学习率开始，将它们存储在类的属性中
    def __init__(self, learning_rate = 1.0, decay = 0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iteration = 0
        # 极小值，防止除以0
        self.epsilon = epsilon


    def pre_update_param(self):
        # 这种衰减的工作原理是取步数和衰减比率并将它们相乘。
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1 / (1 + self.decay * self.iteration))

    # 给一个层对象参数
    def update_param(self, layer):
        if not hasattr(layer, 'dweight_square_sum'):
            layer.dweight_square_sum = np.zeros_like(layer.weight)
            layer.dbias_square_sum = np.zeros_like(layer.bias)
        layer.dweight_square_sum = layer.dweight_square_sum + layer.dweight ** 2
        layer.dbias_square_sum = layer.dbias_square_sum + layer.dbias ** 2
        layer.weight += -self.current_learning_rate * layer.dweight / \
                        ( np.sqrt(layer.dweight_square_sum) + self.epsilon )
        layer.bias += -self.current_learning_rate * layer.dbias / \
                        (np.sqrt(layer.dbias_square_sum) + self.epsilon)

    def post_update_param(self):
        self.iteration += 1

class Optimizer_RMSprop():
    # 初始化方法将接收超参数，从学习率开始，将它们存储在类的属性中
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta = 0.9):
        # 注意：这里的学习率learning_rate = 0.001，不是默认为1
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iteration = 0
        # 极小值，防止除以0
        self.epsilon = epsilon
        self.beta = beta

    def pre_update_param(self):
        # 这种衰减的工作原理是取步数和衰减比率并将它们相乘。
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1 / (1 + self.decay * self.iteration))

    # 给一个层对象参数
    def update_param(self, layer):
        if not hasattr(layer, 'dweight_square_sum'):
            layer.dweight_square_sum = np.zeros_like(layer.weight)
            layer.dbias_square_sum = np.zeros_like(layer.bias)
        layer.dweight_square_sum = self.beta * layer.dweight_square_sum + (1 - self.beta) * layer.dweight ** 2
        layer.dbias_square_sum = self.beta * layer.dbias_square_sum + (1 - self.beta) * layer.dbias ** 2
        layer.weight += -self.current_learning_rate * layer.dweight / \
                        ( np.sqrt(layer.dweight_square_sum) + self.epsilon )
        layer.bias += -self.current_learning_rate * layer.dbias / \
                        (np.sqrt(layer.dbias_square_sum) + self.epsilon)

    def post_update_param(self):
        self.iteration += 1

class Optimizer_Adam():
    # 初始化方法将接收超参数，从学习率开始，将它们存储在类的属性中
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, momentum = 0.9,beta = 0.999):
        # 注意：这里的学习率learning_rate = 0.001，不是默认为1
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iteration = 0
        # 极小值，防止除以0
        self.epsilon = epsilon
        self.beta = beta
        self.momentum = momentum

    def pre_update_param(self):
        # 这种衰减的工作原理是取步数和衰减比率并将它们相乘。
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1 / (1 + self.decay * self.iteration))

    # 给一个层对象参数
    def update_param(self, layer):
        if not hasattr(layer, 'dweight_square_sum') or not hasattr(layer, 'dweight_cumulate'):
            layer.dweight_square_sum = np.zeros_like(layer.weight)
            layer.dbias_square_sum = np.zeros_like(layer.bias)
            layer.dweight_cumulate = np.zeros_like(layer.weight)
            layer.dbias_cumulate = np.zeros_like(layer.bias)
        # 动量
        layer.dweight_cumulate = self.momentum * layer.dweight_cumulate + (1 - self.momentum) * layer.dweight
        layer.dbias_cumulate = self.momentum * layer.dbias_cumulate + (1 - self.momentum) * layer.dbias
        # 微调动量
        layer.dweight_cumulate_modified = layer.dweight_cumulate / (1 - self.momentum ** (self.iteration + 1))
        layer.dbias_cumulate_modified = layer.dbias_cumulate / (1 - self.momentum ** (self.iteration + 1))
        # 平方和
        layer.dweight_square_sum = self.beta * layer.dweight_square_sum + (1 - self.beta) * layer.dweight ** 2
        layer.dbias_square_sum = self.beta * layer.dbias_square_sum + (1 - self.beta) * layer.dbias ** 2
        # 微调平方和
        layer.dweight_square_sum_modified = layer.dweight_square_sum / (1 - self.beta ** (self.iteration + 1))
        layer.dbias_square_sum_modified = layer.dbias_square_sum / (1 - self.beta ** (self.iteration + 1))

        layer.weight += -self.current_learning_rate * layer.dweight_cumulate_modified / \
                        ( np.sqrt(layer.dweight_square_sum_modified) + self.epsilon )
        layer.bias += -self.current_learning_rate * layer.dbias_cumulate_modified / \
                        (np.sqrt(layer.dbias_square_sum_modified) + self.epsilon)

    def post_update_param(self):
        self.iteration += 1

class Dropout():
    def __init__(self, rate):
        # rate是断开连接的概率
        self.rate = 1 - rate

    def forward(self, input, drop_on=True):
        self.input = input
        # 按概率生成一个0、1矩阵
        # 因为1的概率只有rate这么大，就要除以rate偿损失值
        if not drop_on:
            # 如果关上dropout就输出等于输入
            self.output = self.input
            print('这里没有dropout')
            return
        self.mask = np.random.binomial(1, self.rate, size=self.input.shape) / self.rate
        self.output = self.input * self.mask

    def backward(self,dvalue):
        self.dinput = dvalue * self.mask

class Model():
    def __init__(self):
        # 这个属性用来存模型的每层结构
        self.layer = []
        # 先初始化为None，后面会在finalize中判断是否符合softmax+categoricalCrossentropy或sigmiod+binaryCrossentropy
        self.softmax_categoricalCrossentropy = None
        self.sigmoid_binaryCrossentropy = None

    # 用来加入层结构
    def add(self, layer):
        self.layer.append(layer)

    # 用来设置损失loss的类型、优化器等
    # 在星号之后的所有参数都必须作为关键字参数传递，而不能作为位置参数传递
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def evaluate(self, X_val, y_val, *, batch_size=None):
        # 默认只有一个batch
        validation_step = 1
        if batch_size is not None:
            validation_step = len(X_val) // batch_size
            if validation_step * batch_size < len(X_val):  # 如果有余数
                validation_step += 1
        # 清除0
        self.loss.clean_cumulate()
        self.accuracy.clean_cumulate()

        for step in range(validation_step):
            # 没置batch
            if not batch_size:
                X_batch = X_val
                y_batch = y_val
            else:  # 这里有一个很好的性质，当(step+1)*batch_size超过X长度，则自动到最后为止。
                X_batch = X_val[step * batch_size:(step + 1) * batch_size]
                y_batch = y_val[step * batch_size:(step + 1) * batch_size]

            # 输出层的输出
            output = self.forward(X_batch, False)
            # 计算loss
            data_loss, regularization_loss = self.loss.calculate(output, y_batch)
            loss = data_loss + regularization_loss
            # 预测类别或预测值
            prediction = self.output_layer.prediction(output)
            # 计算准确率
            accuracy = self.accuracy.calculate(prediction, y_batch)
        # 平均精度和损失
        validation_accuracy = self.accuracy.calculate_cumulate()
        validation_data_loss, validation_regularizaion_loss = self.loss.calculate_cumulate()
        validation_loss = validation_regularizaion_loss + validation_data_loss
        # 测试输出,输出的是在测试集上的平均表现
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')
        # plt.plot(X_val, y_val)
        # plt.plot(X_val, output)
        # plt.show()


    # 训练模型
    # epochs训练轮数
    # print_every每多少轮输出一次
    def train(self, X, y, *, epochs=1, print_every=1, batch_size=None, validation_data=None):
        # 数据集(默认)分为1个batch
        train_step = 1

        # 非默认情况
        if batch_size is not None:
            train_step = len(X) // batch_size
            if train_step * batch_size < len(X): # 如果有余数
                train_step += 1

        # 注意：validation_data需要输入一个元组，包括X、y
        for epoch in range(1, epochs+1):
            print(f'epoch:{epoch}')
            # 清累积
            self.loss.clean_cumulate()
            self.accuracy.clean_cumulate()

            for step in range(train_step):
                # 没置batch
                if not batch_size:
                    X_batch = X
                    y_batch = y
                else: # 这里有一个很好的性质，当(step+1)*batch_size超过X长度，则自动到最后为止。
                    X_batch = X[step*batch_size:(step+1)*batch_size]
                    y_batch = y[step*batch_size:(step+1)*batch_size]

                # 前向传播
                output = self.forward(X_batch)
                # 计算损失
                data_loss, regularization_loss = self.loss.calculate(output, y_batch, add_regular_loss=True)
                # 总loss
                loss = data_loss + regularization_loss
                # 计算预测值或预测类别
                prediction = self.output_layer.prediction(output)
                # 计算准确率
                accuracy = self.accuracy.calculate(prediction, y_batch)

                # 反向传播
                self.backward(output, y_batch)

                # 优化器进行优化
                self.optimizer.pre_update_param()
                for layer in self.trainable_layer:
                    self.optimizer.update_param(layer)
                self.optimizer.post_update_param()

                # step中打印的是每次的真实值
                if not step % print_every or step == train_step - 1:
                    print(f'step: {step}, ' +
                        f'acc: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f} (' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularization_loss:.3f}), ' +
                        f'lr: {self.optimizer.current_learning_rate}')

            # 让epoch输出，输出每次epoch的平均值
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_cumulate(add_regularization=True)
            epoch_loss = epoch_regularization_loss + epoch_data_loss
            epoch_accuracy = self.accuracy.calculate_cumulate()
            # 输出信息，输出每次epoch的平均值
            print(f'training {epoch}, ' +
                f'acc: {epoch_accuracy:.3f}, ' +
                f'loss: {epoch_loss:.3f} (' +
                f'data_loss: {epoch_data_loss:.3f}, ' +
                f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                f'lr: {self.optimizer.current_learning_rate}')


            if validation_data is not None:
                self.evaluate(*validation_data,batch_size=batch_size)

    ## 在该方法内实现模型的定型
    # 1.确定不同层之间的前后次序
    # 2.确定Dense层
    # 3.将Dense层传入loss对象中，以计算正则损失
    # 4.判断是否符合softmax+categoricalCrossentropy或sigmiod+binaryCrossentropy
    def finalize(self):
        # 创建输入层
        self.input_layer = Layer_Input()
        # 模型层数，不包括输入层、loss层
        layer_num = len(self.layer)
        # 存放Dense层（有参数可以学习）
        self.trainable_layer = []

        # 循环设置层间关系
        for i in range(layer_num):
            if i == 0:
                # 第一层Dense,它的前一层是input_layer
                self.layer[i].pre = self.input_layer
                self.layer[i].next = self.layer[i + 1]
            elif i == layer_num-1:
                # 最后一个Dense，它是后一层是loss
                self.layer[i].pre = self.layer[i - 1]
                self.layer[i].next = self.loss
                # 在最后一层标记一下所用的输出层是什么Activation存在Model的属性中
                self.output_layer = self.layer[i]
            else:
                self.layer[i].pre = self.layer[i-1]
                self.layer[i].next = self.layer[i+1]

            if hasattr(self.layer[i], 'weight'):
                # 如果当前层有'weight'属性，说是当前层是Dense层
                # 该层是可以训练的
                self.trainable_layer.append(self.layer[i])

        # 把Dense层告诉loss对象
        self.loss.save_trainable_layer(self.trainable_layer)
        # 判断是否符合softmax+categoricalCrossentropy或sigmiod+binaryCrossentropy
        if isinstance(self.layer[-1], Activation_Softmax) and \
                isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_categoricalCrossentropy = Activation_Softmax_Loss_CategoricalCrossentropy()

        if isinstance(self.layer[-1], Activation_Sigmoid) and \
                isinstance(self.loss, Loss_BinaryCrossentropy):
            self.sigmoid_binaryCrossentropy = Activation_Sigmoid_Loss_BinaryCrossentropy()

    # 前向传播
    # 该方法将在train方法中调用（训练过程将调用很多种方法，forward中是其中一个）
    def forward(self, input, dropout=True):
        self.input_layer.forward(input)
        for layer in self.layer:
            if isinstance(layer,Dropout) and (not dropout):
                layer.forward(layer.pre.output,dropout)
            else:
                layer.forward(layer.pre.output)

        # 这里的layer是最后一层的activation
        return layer.output

    def backward(self, output, y_true):
        if self.softmax_categoricalCrossentropy:
            self.softmax_categoricalCrossentropy.backward(output, y_true)
            # 最后一层是softmax,不调用backward求dinput,
            # 因为softmax_categoricalCrossentropy已经算好
            self.layer[-1].dinput = self.softmax_categoricalCrossentropy.dinput
            # 注意：这里循环不包含最后一层（softmax）
            for layer in reversed(self.layer[:-1]):
                layer.backward(layer.next.dinput)
            return
        if self.sigmoid_binaryCrossentropy:
            self.sigmoid_binaryCrossentropy.backward(output, y_true)
            # 最后一层是sigmoid,不调用backward求dinput,
            # 因为softmax_categoricalCrossentropy已经算好
            self.layer[-1].dinput = self.sigmoid_binaryCrossentropy.dinput
            # 注意：这里循环不包含最后一层（softmax）
            for layer in reversed(self.layer[:-1]):
                layer.backward(layer.next.dinput)
            return

        self.loss.backward(output, y_true)
        # 注意：这里用的不是self.trainable_layer
        for layer in reversed(self.layer):
            layer.backward(layer.next.dinput)

class Accuracy:
    # 计算准确率
    def calculate(self, prediction, y_true):
        # 获得比较结果
        comparision = self.compare(prediction, y_true)
        # 计算准确率
        accuracy = np.mean(comparision)
        # 加入了累积精度属性
        self.cumulate_dataloss += np.sum(comparision)
        self.cumulate_num += len(comparision)

        return accuracy

    def calculate_cumulate(self):
        # 平均精度
        accuracy = self.cumulate_dataloss / self.cumulate_num
        return accuracy

    def clean_cumulate(self):
        self.cumulate_dataloss = 0
        self.cumulate_num = 0

class Accuracy_Regresion(Accuracy):
    def __init__(self):
        # 创建一个属性，保存精度
        # 因为对于Regresion，要自己先创建一个精度标准
        self.precision = None

    def compare(self, precision, y_true):
        if self.precision is None:
            self.precision = np.std(y_true) / 250
        return np.abs(precision - y_true) < self.precision

class Accuracy_Classification(Accuracy):
    def __init__(self):
        pass

    def compare(self, precision, y_true):
        # onehot编码
        if len(y_true.shape) == 2:
            # 改成单个类别
            y_true = np.argmax(y_true,axis=1) #此时是行向量，可能用keepdims=保持矩阵
        # 注意：prediction是一个矩阵，y_true是一个向量1xa
        # 当矩阵是ax1时，会错误产生广播
        # 非常重要，我以为是模型代码错了一天的bug，
        # 最后发现可能只是正确率证算错误了
        y_true = y_true.reshape(-1, 1)
        compare = (precision == y_true) * 1
        return compare




nnfs.init()#默认随机种子为0，每次运行同样的数据
X, y = spiral_data(samples=1000, classes=3)
key = np.array(range(X.shape[0]))
np.random.shuffle(key)
X = X[key]
y = y[key]
X_test, y_test = spiral_data(samples=100, classes=3)
# print(X[:5])
# print(X_test[:5])

model = Model()
model.add(Layer_Dense(2,64,weight_L2=5e-4,bias_L2=5e-4))#,weight_L2=5e-4,bias_L2=5e-4
model.add(Activation_ReLu())

model.add(Layer_Dense(64,3))
model.add(Activation_Softmax())
model.set(loss=Loss_CategoricalCrossentropy(),
          optimizer=Optimizer_Adam(decay=5e-7),
          accuracy=Accuracy_Classification())


model.finalize()

model.train(X,y,batch_size=10,validation_data=(X_test,y_test),epochs=1000,print_every=100)
model.evaluate(X_test,y_test,batch_size=10)







