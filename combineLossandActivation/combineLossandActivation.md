# combine the Loss and Activation Function

## 一、内容

在之前内容中已经实现了了Categorical Cross-Entropy函数和Softmax激活函数，但是还可以进一步来加速计算。这部分是因为两个函数的导数结合起来使整个代码实现更简单、更快。除此之外，Binary Cross-Entropy loss和Sigmoid也能结合。

## 二、Categorical Cross-Entropy loss and Softmax activation

### **公式**

$$
L_i=-\sum\limits_jy_{i,j}log(\hat y_{i,j})
$$

> 在Backpropagation的Softmax部分讲到了$\frac{\partial S_{i,j}}{\partial z_{i,k}}$的计算，且$\hat y_{i,j}=S_{i,j}$，所以有：

![image-20230809092620943](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308090926027.png)

![image-20230809092641430](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308090926463.png)

> 在Backpropagation的Categorical Cross-Entropy loss部分讲到了：

![image-20230809093259100](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308090932146.png)

> 综上有：

![image-20230809093342583](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308090933626.png)

> **注意：这里的$z$是Softmax的input，$L$是Categorical Cross-Entropy的output**

### **实现**

```python
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
        if len(y_true.shape) == 2: # onehot编码
            # 直接套公式
            self.dinput = y_pred - y_true
        elif len(y_true.shape) == 1: # 只有一个类别
            self.dinput = y_pred.copy()
            # 需将每一行中y_true类别（索引）中的-1，其它-0（不操作）
            self.dinput[range(n_sample), y_true] -= 1
        # 每个样本除以n_sample，因为在优化的过程中要对样本求和
        self.dinput = self.dinput / n_sample
```

### **实例**

```python
##########################################
softmax_outputs = np.array([[0.7, 0.1, 0.2],[0.1, 0.5, 0.4],[0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])
softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinput

activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_CategoricalCrossentropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinput)
dvalues2 = activation.dinput

print('Gradients: combined loss and activation:')
print(dvalues1)
print('Gradients: separate loss and activation:')
print(dvalues2)
###################################################
```

![image-20230809123920168](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308091239219.png)

> 将Activation和loss分开，或都合并都实现了相同的结果。

```python
def f1():
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinput
def f2():
    activation = Activation_Softmax()
    activation.output = softmax_outputs
    loss = Loss_CategoricalCrossentropy()
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinput)
    dvalues2 = activation.dinput

t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print(t2/t1)
```

![image-20230809124516686](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308091245718.png)

> 可以看到，当两种实现方法重复10000次以后，所用时间接近4倍。

## 三、Sigmoid and Binary Cross-Entropy Loss

> **这部分内容在书中并没有， 是我自己根据理解后，推导公式和编程实现的，并不代表完全正确。将在更深入学习后勘误。**

### **公式**

> 参照Sigmoid和Binary Cross-Entropy的求代公式有（第个样本下标$i$，省去）：

$$
\frac{\partial L}{\partial \hat y_j} = -\frac{1}{J}(\frac {\partial y_j}{\partial \hat y_j} - \frac {1-\partial y_j}{1-\partial \hat y_j})
$$

$$
\frac{\partial \sigma_j}{\partial z_j} = \sigma_j(1-\sigma_j)
$$

> 因为，Sigmoid的输出$\sigma$就是Binary Cross-Entropy的输入$\hat y$，写成矩阵形式，$\frac{\partial L}{\partial z}$和$\frac{\partial L}{\partial \hat y}$是行向量，$\frac{\partial \sigma}{\partial z}$是对角方阵。

$$
\frac{\partial L}{\partial z}=\frac{\partial L}{\partial \hat y}\frac{\partial \sigma}{\partial z}
$$

> 对每个标量进行计算有：

$$
\frac{\partial L}{\partial z_j}=\frac{\partial L}{\partial \hat y_j}\frac{\partial \sigma_j}{\partial z_j}=\frac{\partial L}{\partial \hat y_j}\frac{\partial \hat y_j}{\partial z_j}= -\frac{1}{J}(\frac {\partial y_j}{\partial \hat y_j} - \frac {1-\partial y_j}{1-\partial \hat y_j})\hat y_j(1-\hat y_j)=\frac{\hat y_j-y_j}{J}
$$

### **实现**

```python
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
        self.dinput = ( y_pred - y_true ) / J

        # 优化时要将所有样本相加，为了梯度与样本数量无关，这里除以样本数
        self.dinput /= n_sample
```

> **注意看注释，非常重要**

### **实例**

```python
##########################################
# 数据
X, y = spiral_data(samples=100,classes=2)
print(X.shape)
print(X[:5])

# 两层Dense，一层ReLu
dense1 = Layer_Dense(2,4)
dense2 = Layer_Dense(4,1)
activation1 = Activation_ReLu()

# 前向传播
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
sigmoid_in = dense2.output
print(sigmoid_in[:5])
####
##
sigmoid_loss = Activation_Sigmoid_Loss_BinaryCrossentropy()
dataloss1 = sigmoid_loss.forward(sigmoid_in, y)
##
activation2 = Activation_Sigmoid()
loss = Loss_BinaryCrossentropy()
activation2.forward(sigmoid_in)
dataloss2 = loss.calculate(activation2.output, y)
##
####

# 反向传播
####
##
sigmoid_loss.backward(sigmoid_loss.output, y)
dinput1 = sigmoid_loss.dinput
##
loss.backward(activation2.output, y)
activation2.backward(loss.dinput)
dinput2 = activation2.dinput

print('Gradients: combined loss and activation:')
print(dataloss1)
print(dinput1.shape)
print(dinput1[50:55])

print('Gradients: separate loss and activation:')
print(dataloss2)
print(dinput2.shape)
print(dinput2[50:55])
```



![](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308091744550.png)

> 两种实现方法计算得到的loss和梯度是一样的。

```python
def f1():
    sigmoid_loss.backward(sigmoid_loss.output, y)
    dinput1 = sigmoid_loss.dinput
def f2():
    loss.backward(activation2.output, y)
    activation2.backward(loss.dinput)
    dinput2 = activation2.dinput

t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print(t2/t1)
```

![image-20230809175814647](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308091758689.png)

>  两种方法重复10000次，运行时间相差6倍。
