# Activation

## 一、内容

在本部分将实现常用的Activation Function，例如：ReLu（Rectified Linear units Function）、Softmax、Sigmoid。本部分只实现forward method，反向传播将在后续加入

## 二、代码

### 一、Sigmoid

1. 公式

   其中$z_{i,j}$表示这个激活函数的输入，$\sigma_{i,j}$表示单个输出值。索引$i$表示当前样本，索引$j$ 表示当前样本中的当前输出。$\sigma_{i,j}$可理解成对第$j$对类别，例如猫狗分类中狗类别的confidence(置信度)。当然，一个模型可能要对多对类别分类，例如：高矮、胖瘦等。Sigmoid用于二分类
   $$
   \sigma_{i,j}=\frac{1}{1+e^{-z_{i,j}}}
   $$

2. 实现

   ~~~py
   class Activation_Softmax:
       def __init__(self):
           pass
   
       def forward(self, input):
           # input的大小是nx1，n是Activation输入的sample数量，每个sample只有一个维度。
           # 所以前一个hidden layer必须是Layer_Dense(n, 1)
           self.output = 1 / ( 1 + np.exp(-input) )
   ~~~

   

3. 实例

   ~~~py
   # 生成数据
   X, y = spiral_data(samples=100, classes=2)
   # 构建一个含三个神经元的Dense层实例
   dense = Layer_Dense(2,1)
   # 构建Softmax激活函数
   activation1 = Activation_Softmax()
   
   # 前向传播
   dense.forward(X)
   activation1.forward(dense.output)
   # 输出结果
   print(activation1.output[:5])
   ~~~

   ![image-20230807190723399](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308071907435.png)

> 输出是nx1大小，表示二分类别的confidence(置信度)

### 二、ReLu

1. 公式
   $$
   y=\begin{cases}x,x > 0\\0,x \le 0\end{cases}
   $$
   

2. 实现

   ~~~python
   class Activation_ReLu:
       def __init__(self):
           pass
   
       def forward(self,input):
           self.output = np.maximum(0,input)
   ~~~

   

3. 实例

   ~~~python
   # 生成数据
   X, y = spiral_data(samples=100, classes=3)
   # 构建一个含三个神经元的Dense层实例
   dense = Layer_Dense(2,3)
   # 构建Softmax激活函数
   activation1 = Activation_ReLu()
   
   # 前向传播
   dense.forward(X)
   activation1.forward(dense.output)
   # 输出结果
   print(activation1.output[:5])
   ~~~

   ![image-20230807193322759](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308071933795.png)

### 三、Softmax

​		Softmax函数是一种将j个实数向量转换为j个可能结果的概率分布的函数。索引i表示当前样本，索引j表示当前样本中的当前输出，$S_{i,j}$表示j个可能结果的概率。

  1. 公式
     $$
     S_{i,j}=\frac{e^{z_{i,j}}}{\sum\limits_{l=1}^L{e^{z_{i,l}}}}
     $$
     

  2. 实现

     softmax函数对输入值非常敏感，而且很容易产生极端的概率分布。这可能会导致模型过度自信地预测某个类别，而忽略了其他类别的可能性。为了避免这种情况，我们可以在进行指数运算之前，从输入值中减去最大值。这样做不会改变softmax函数的结果，因为分子和分母都会被同一个常数除以。但是，这样做可以使输入值更小，从而避免指数运算产生过大的数字。

     ~~~python
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
     ~~~

     

  3. 实例

     ```python
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
     ```

​	![image-20230807202215791](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308072022825.png)



