# Loss

## 一、内容

在本部分将实现的Loss，CategoricalCrossentropy类（继承了Loss类）。本部分只实现forward method，反向传播将在后续加入。

## 二、代码

### 一、Loss父类

  1. 实现

     ```python
     class Loss:
         def __init__(self):
             pass
     
         # 统一通过调用calculate方法计算损失
         def calculate(self, prediction, y):
             # 对于不同的损失函数，通过继承Loss父类，并实现不同的forward方法。
             data_loss = np.mean( self.forward(prediction, y) )
             # 注意，这里计算得到的loss不作为类属性储存，而是直接通过return返回
             return data_loss
     ```

### 二、CategoricalCrossentropy类

  1. 公式
     $$
     L_i=-\sum\limits_jy_{i,j}log(\hat{y}_{i,j})
     $$

     > 当预测属于A、B、C三个类的概率分别是0.7，0.1、0.2，其实类别为A，测$L_i$计算如下。其中i表示对第i个sample计算得到的loss

     ![](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308072055997.png)

  2. 实现

     ```python
     class Loss_CategoricalCrossentropy(Loss):
         def __init__(self):
             pass
     
         def forward(self, y_pred, y_ture):
             # 多少个样本
             n_sample = len(y_ture)
     
             # 为了防止log(0)，所以以1e-7为左边界
             # 另一个问题是将置信度向1移动，即使是非常小的值，
             # 为了防止偏移，右边界为1 - 1e-7
             y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
     
             loss = - np.log(y_pred)
             if len(y_ture.shape) == 2:# 标签是onehot的编码
                 loss = loss * y_ture
             elif len(y_ture.shape) == 1:# 只有一个类别标签
                 # 注意loss = loss[:, y_ture]是不一样的，这样会返回一个矩阵
                 loss = loss[range(n_sample), y_ture]
     
             return loss
     ```

  3. 实例

     ```python
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
     ```

![image-20230807220820346](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308072208379.png)

