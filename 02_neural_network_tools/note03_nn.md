# Pytorch搭建神经网络（3）利用 `nn` 便捷搭建神经网络

基于[《深度学习框架 Pytorch 入门与实践》陈云](https://book.douban.com/subject/27624483/) 

参考 [Github 的 pytorch-book 项目](https://github.com/chenyuntc/pytorch-book)

参考 [GitHub 的 pytorch-handbook 项目](https://github.com/zergtant/pytorch-handbook)

---

本章主要讲解如何使用 Pytorch 实现深度学习/神经网络里的结构和功能，关注实践，理论较少。

`nn` 模块是 Pytorch 提供的神经网络模块，可以快速便捷地搭建神经网络或神经网络里的各个层（layer）。

# 1 利用 nn.Module 实现全连接层和多层感知机

在实际应用中，我们往往继承类 `torch.nn.Module` ，然后便携自己的网络层。下面以实现全连接层作为简单引入。

## 1.1 全连接层

全连接层可以简单理解为一个线性层，它接受输入的张量 `x.shape = (?, in_features)` 并返回结果 `y.shape = (?, out_features)` ，利用的就是简单的线性组合。
$$
y = W x + b
$$
其中 $W \in \R^{\text{in\_features}\times\text{out\_features}}$ 而 $b \in \R^{\text{out\_features}}$。

> 注意：此处的乘是类似矩阵乘法，而【不是逐元素相乘】

```python
""" 定义线性层 Linear 用来计算 y = W x + b """
class Linear(nn.Module):  # 继承 nn.Module
    def __init__(self, in_features, out_features):
        # in_features 输入的形状，out_features 输出的形状
        super().__init__()  # 等价于 nn.Module.__init__(self)
        # nn.Parameter 指定需要网络学习的参数
        self.W = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))

    # 前向传播
    def forward(self, x):
        # 计算 y = xW + b : 利用了广播机制，b 会复制成 y 一般大小，即 (out_features,)
        y = x @ self.W + self.b  # @ 代表矩阵乘法
        return y
```

- 需要使用 `super()` 方法调用父类的 `__init__()` 方法。或者直接使用 `nn.Module.__init__(self)`
- 注意在定义自己的 `__init__()` 时，需要声明参数。例如这里的 `in_features` 和 `out_features` 
- `in_features` 和 `out_features` 指定输入输出的形状
- `nn.Parameter()` 指定网络需要学习的参数，用来告诉网络之后需要更新的对象
- 注意参数的形状，需要满足 `(?, in_features) @ (in_features, out_features) -> (?, out_features)` 这类似矩阵乘法，不过此处是张量

调用上述定义的线性层/全连接层，检查维度是否正确

```python
# 调用上述定义的线性层/全连接层，检查维度
linear_layer = Linear(in_features=4, out_features=3)
inputs = torch.randn(2, 4)
outputs = linear_layer(inputs)
print(outputs.shape)
# torch.Size([2, 3]) : (2, 4) @ (4, 3) -> (2, 3)
```

使用 `.named_parameters()` 方法检查参数 `W, b`

```python
for name, parameter in linear_layer.named_parameters():
    print("1. It is parameter: {}".format(name))
    print("2.", parameter)
    print("3. The shape is: {}\n".format(parameter.shape))
```

```python
# 上述检查参数的返回结果
[Out]:  1. It is parameter: W
        2. Parameter containing:
        tensor([[ 1.1711,  0.4335, -1.7343],
                [-1.3360,  0.8871,  0.7680],
                [ 0.0571,  0.2240,  0.5520],
                [-0.5788,  0.0177,  0.1318]], requires_grad=True)
        3. The shape is: torch.Size([4, 3])

        1. It is parameter: b
        2. Parameter containing:
        tensor([ 1.0198, -0.4468,  0.4520], requires_grad=True)
        3. The shape is: torch.Size([3])
```

## 1.2 多层感知机

由多个线性层/全连接层通过某些激活函数构成的网络，称为多层感知机。

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739156093597.png)

根据上图的网络结构搭建多层感知机：

```python
class MultiPerceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        # 新增参数：隐藏层神经元个数（形状）
        super().__init__()
        # 直接使用之前定义的线性层/全连接层 Linear
        self.layer1 = Linear(in_features, hidden_features) 
        self.layer2 = Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)  # 使用激活函数，增加非线性因素（此处是逐个元素计算）
        y = self.layer2(x)
        return y
```

- 之前定义的层 Layer 可以在后续重复使用
- 注意传入参数，用以确认形状

调用上述定义的多层感知机，检查维度是否正确

```python
# 检查维度
mlp = MultiPerceptron(3, 4, 1)
inputs = torch.randn(2, 3)
outputs = mlp(inputs)
print(outputs.shape)
# torch.Size([2, 1]) ： (2, 3) @ (3, 4) @ (4, 1) -> (2, 1)
```

检查参数

```python
# 检查参数
for name, param in mlp.named_parameters():
    print(name, param.size())

# layer1.W torch.Size([3, 4])
# layer1.b torch.Size([4])
# layer2.W torch.Size([4, 1])
# layer2.b torch.Size([1])
```

> 【注意输入形状】输入的形状一般为 `(?, in_features)` 其中 `?` 一般为 `batch_size` 即样本集个数。
>
> 当输入单一数据时，即只输入一个样本时，需要扩展维度，利用[第一章](https://blog.iskage.online/posts/652f5539.html#8-3-%E7%BB%B4%E5%BA%A6%E5%8E%8B%E7%BC%A9%E3%80%81%E6%89%A9%E5%B1%95%E3%80%81%E6%8B%BC%E6%8E%A5-squeeze-unsqueeze-cat)介绍的 `unsqueeze()` 函数。向前扩展一个维度 `tensor.unsqueeze(0)` ，例如：

```python
# batch_size = 1
x = torch.randn(3)
x.unsqueeze_(0)  # 需要向前扩展 1 个维度 （`_` 表示 inplace 操作，直接替换 x）
y = mlp(x)
print(y.shape)  # 正确 torch.Size([1, 1])
```



总结：Pytorch 的 nn 封装了非常多网络层，可以直接前往[官方文档](https://pytorch.org/docs/stable/nn.html)查看。下面介绍常见的网络层。

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739157306136.png)



# 2 常见神经网络层

## 2.1 以图像处理为例

图像相关层主要包括：卷积层 `Conv`、池化层 `Pool` 等。往往还有不同维度图像处理的分类，同时池化方法也有最大池化、均值池化等。

> 建议先学习卷积的原理，参考课程
>
> 中文，更专业：b站 [【19 卷积层【动手学深度学习v2】】](https://www.bilibili.com/video/BV1L64y1m7Nh/?share_source=copy_web&vd_source=67ce2d561f3b6dc9d7cff375959101a2)
>
> 中文，更易懂：b站 [【【子豪兄】精讲CS231N斯坦福计算机视觉公开课（2020最新）】](https://www.bilibili.com/video/BV1K7411W7So/?p=5&share_source=copy_web&vd_source=67ce2d561f3b6dc9d7cff375959101a2)
>
> 英文，更专业：[cs231n](https://cs231n.stanford.edu/)

### 2.1.1 卷积层

图像处理相关的神经网络层，最最重要的就是卷积层。以 `Conv2d` 为例，介绍里面的参数和使用方法。

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
```

**参数**

```python
# in_channels: 输入
- in_channels (int) – 图片的通道数，例如RGB图片就是 3 通道，灰度图只有 1 通道

# out_channels: 输出
- out_channels (int) – 输出结果的通道数

# kernel_size: 卷积核的大小
- kernel_size (int or tuple) – 卷积核的大小，只需输入T 则会自动生成一个 (T, T, channels) 大小的卷积核

# stride: 步数
- stride (int or tuple, optional) – 卷积核每次移动的步数，默认为 1

# padding: 填充层数
- padding (int or tuple, optional) – 填充层数，用以维持图片大小的参数，默认为 0 

# padding_mode: 填充方式
- padding_mode (string, optional) – 填充方式，一般默认即可，有 'zeros', 'reflect', 'replicate' or 'circular' 多种选择，默认为 'zeros'

# dilation: 卷积核中元素的对应位置
- dilation (int or tuple, optional) – 默认为 1

# groups: 可选
- groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1

# bias: 可选
- bias (bool, optional) – 是否增加偏倚项，默认为 True : If True, adds a learnable bias to the output. Default: True
```

> 如果希望卷积后，通道变多，但尺寸不变，则需要填充 `padding` ，公式

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/image-20240731%E4%B8%8A%E5%8D%88110036714.png)

卷积过程的动画展示可参考 [https://github.com/vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic)

原理简单理解【卷积】

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/image-20240731%E4%B8%8A%E5%8D%88101819072.png)

原理简单理解【padding】

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/image-20240731%E4%B8%8A%E5%8D%88102514146.png)

### 2.1.2 代码：使用卷积层

导入库，进行图片处理

```python
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

to_tensor = ToTensor()  # img -> Tensor
to_pil = ToPILImage()  # Tensor -> PIL
```

选择一张图片（图源网络）点此下载 [lena's photo](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/lena.png)

```python
example = Image.open('imgs/lena.png')
example # 可视化输出
```

![lena](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/lena.png)

查看输入图片形状

```python
example = to_tensor(example).unsqueeze(0)  # 补充 batch_size
print("Input Size:",example.size()) # 查看 input 维度
# Input Size: torch.Size([1, 1, 200, 200])
```

查看卷积后输出图片形状

```python
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

out = conv(example)
print("Output Size:",out.size())
# Output Size: torch.Size([1, 1, 198, 198])
# 198 = (200 + 2 * 0 - 3 )/1 + 1 = 198
```

以图片形式输出

```python
to_pil(out.data.squeeze(0))  # 去除 batch_size 转换为图片输出
```

![lena_conv](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/lena_conv.png)

> 拓展：指定卷积核

指定卷积核可以达到不同的效果

```python
# 拓展：指定卷积核
kernel = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.]
], dtype=torch.float32)  # 提取竖直边缘特征

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=0, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)  # (batch_size, in_channels, height, width)

out = conv(example)
print("Output Size:", out.size())  # torch.Size([1, 1, 198, 198])

to_pil(out.data.squeeze(0))  # 去除 batch_size 转换为图片输出
```

![lena_conv_with_certain_kernel](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/lena_kernel.png)

### 2.1.3 池化层

池化层模糊选取某些特征，某些意义上可以防止过拟合。以最大池化为例，他选取范围内最大值替换整个范围。

```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

**参数**

```python
# 取最大值的窗口
- kernel_size – the size of the window to take a max over

# 横向纵向的步长，default = kernel_size
- stride – the stride of the window. Default value is kernel_size

# 补充图像边缘
- padding – implicit zero padding to be added on both sides

# 空洞
- dilation – a parameter that controls the stride of elements in the window

- return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later

# floor向下取整 ceil向上取整，例如ceil_mode = True，保留超出部分
- ceil_mode – when True, will use ceil instead of floor to compute the output shape
```

结合下图例理解最大池化原理

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/image-20240804%E4%B8%8B%E5%8D%8832249522.png)

代码实现上述案例，进行验证

```python
from torch.nn import MaxPool2d

inputs = torch.tensor([
    [1, 2, 0, 2, 1, ],
    [0, 1, 3, 1, 1, ],
    [1, 2, 1, 0, 0, ],
    [5, 2, 3, 1, 1, ],
    [2, 1, 0, 1, 1, ],
], dtype=torch.float)

# 1 batch_size，1 通道，5x5 大小，-1 表示自动计算
inputs = torch.reshape(inputs, (-1, 1, 5, 5))

# 神经网络
max_pool = MaxPool2d(kernel_size=3, ceil_mode=True)

output = max_pool(inputs)
print(output)
```

```python
[Out]: tensor([[[[3., 2.],
                 [5., 1.]]]])  # 确实与手算结果相同
```

> 根据池化原理，只是做了简单的取值替换，故【没有可学习的参数】

```python
list(max_pool.parameters())
[Out]: []
```

### 2.1.4 代码：使用池化层

```python
out = max_pool(example)
to_pil(out.data.squeeze(0)) # 输出池化后的lena
```

![lena_max_pool](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/lena_pool.png)

容易发现，经过最大池化后，图片变小，变模糊。

```python
out.shape
# torch.Size([1, 1, 67, 67])
```

## 2.2 其他常见层

### 2.2.1 线性层/全连接层

`nn.Linear` 层提供了类似计算 $y = Wx+b$ 的功能

```python
# 线性层
inputs = torch.randn(2, 3)
linear_out = nn.Linear(3, 4)
out = linear_out(inputs)
out.shape
# torch.Size([2, 4]) : (2, 3) @ (3, 4) -> (2, 4) where 2 is batch_size
```

更多可参见 [Pytorch 搭建神经网络（2）网络搭建 - 线性层](https://blog.iskage.online/posts/ae1c954d.html#1-Linear)

### 2.2.2 批量归一化层

`nn.BatchNorm1d` 层提供对 1 维数据进行归一化，填入的参数为特征数（例如上一个输出的维度）

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 512)
        self.bn = nn.BatchNorm1d(512)  # 全连接层后接BatchNorm1d
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
```

更多可参见 [Pytorch 搭建神经网络（2）网络搭建 - 正则化层](https://blog.iskage.online/posts/ae1c954d.html#7-%E6%AD%A3%E5%88%99%E5%8C%96%E5%B1%82)

### 2.2.3 Dropout 层

`nn.Dropout` 层用于防止过拟合，按照概率遗弃一些神经元

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(0.5)  # 以 0.5 的概率遗弃
        self.fc2 = nn.Linear(128, 10)
	def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## 2.3 循环神经网络

PyTorch 中提供了最常用的三种循环神经网络：`RNN`、`LSTM` 和 `GRU` 。

推荐学习 [《动手学深度学习》](https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/index.html) 中关于循环神经网络的知识，十分详细。也可结合李沐老师的讲解[b站连接](【54 循环神经网络 RNN【动手学深度学习v2】】 https://www.bilibili.com/video/BV1D64y1z7CA/?share_source=copy_web&vd_source=67ce2d561f3b6dc9d7cff375959101a2)



## 3 激活函数

激活函数可以为模型加入非线性性。

这部分可以参见 [Pytorch 搭建神经网络（2）网络搭建 - 激活函数](https://blog.iskage.online/posts/ae1c954d.html#6-%E9%9D%9E%E7%BA%BF%E6%80%A7%E6%BF%80%E6%B4%BB%EF%BC%88%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%EF%BC%89)



## 4 前馈传播网络的便捷构建

上述的网络结构均为：前一层的输出是下一层的输入。这样的网络结构称为**前馈传播网络**（Feedforward Neural Network，FFN）。

针对这样的网络结构，可以使用 `ModuleList` 和 `Sequential` 来组合各个层。

### 4.1 Sequential

使用 `Sequential` 的三种方法：将卷积层、归一化层和激活函数层组合成一个网络

```python
# 法一
net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3, 3, 3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('relu', nn.ReLU())

print('net1:', net1)
# net1: Sequential(
#   (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
#   (batchnorm): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU()
)
```

```python
# 法二
net2 = nn.Sequential(
    nn.Conv2d(3, 3, 3),
    nn.BatchNorm2d(3),
    nn.ReLU()
)

print('net2:', net2)
# net2: Sequential(
#   (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
#   (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
# )
```

```python
# 法三
from collections import OrderedDict

net3 = nn.Sequential(OrderedDict([
    ('conv', nn.Conv2d(3, 3, 3)),
    ('batchnorm', nn.BatchNorm2d(3)),
    ('relu', nn.ReLU())
]))

print('net3:', net3)
# net3: Sequential(
#   (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
#   (batchnorm): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU()
# )
```

- 可以根据名字和序号取出对应的层

```python
net1.conv
# Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))

net2[1]
# BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

net3.relu
# ReLU()
```

### 4.2 ModuleList

使用 `nn.ModuleList` 连接三个层

```python
model_list = nn.ModuleList([nn.Linear(3,4), nn.ReLU(), nn.Linear(4,2)])
inputs = torch.randn(1, 3)
for model in model_list:
    inputs = model(inputs)  # 一步一步执行，相当于前向传播 forward
inputs.shape
# torch.Size([1, 2])
```

> 不可以直接调用 `modellist(inputs)` ，因为没有定义前向传播

```python
inputs = torch.randn(1, 3)
output = modellist(inputs)  # 报错，没有定义 forward 函数

# NotImplementedError: Module [ModuleList] is missing the required "forward" function
```

> 【不能直接使用 list 类型】必须使用 `nn.ModuleList` 连接各个层，直接使用 `list` 类型是无法继承 `nn.Module` 从而无法被识别



## 5 损失函数

Pytorch 提供简单计算损失的函数，例如均方误差、交叉熵损失等。

- 均方误差损失 `nn.MSELoss()`

```python
# 生成预测值和真实值
y_pred = torch.randn(4, 1)
y_real = torch.randn(4).squeeze(-1)  # 将 y_real 的形状调整为 (4, 1)

# 初始化 MSE 损失函数
mse = nn.MSELoss()

# 计算损失
loss = mse(y_pred, y_real)

print(loss)  # tensor(1.2719)
```

- 交叉熵损失 `nn.CrossEntropyLoss()`

```python
# batch_size=4，即这一组共 4 个样本，类别为 2
score = torch.randn(4, 2)  # 4 个样本，每个样本对应 2 个数值，代表属于第 0 or 1 类的概率
# 假设 4 个样本的真实类为：1, 0, 1, 1 
label = torch.Tensor([1, 0, 1, 1]).long()  # label 必须为 LongTensor

# 交叉熵损失 CrossEntropyLoss （常用与计算分类问题的损失）
criterion = nn.CrossEntropyLoss()
loss = criterion(score, label)

print(loss)  # tensor(0.5944)
```



## 6 nn.functional 模块

使用 `nn.Module` 实现的层是一个特殊的类，其由 `class layer(nn.Module)` 定义，会自动提取可学习的参数；使用`nn.functional`实现的层更像是纯函数，由`def function(input)`定义。

也就是说，当这一层无需学习参数时，使用 `nn.functional` 是合理的。

### 6.1 使用 nn.functional 的函数

以 `nn.functional.linear()` 为例，其他函数可参考官网 [https://pytorch.org/docs/stable/nn.functional.html](https://pytorch.org/docs/stable/nn.functional.html)

```python
torch.nn.functional.linear(input, weight, bias=None) -> Tensor
```

**参数**

```python
- input: (batch_size, in_features)
输入值，需要为 tensor

- weight: (in_features, out_features)
权重，需要为 tensor

- bias: (out_features) or None
偏倚，需要为 tensor，或者为空
```

```python
inputs = torch.randn(2, 3)

# 1. 使用 nn.Module
model = nn.Linear(3, 4)
output1 = model(inputs)

# 2. 使用 nn.functional
output2 = nn.functional.linear(inputs, model.weight, model.bias)  # 这里使用与 1 相同的参数

print(output1)
print(output2)
# 二者值完全一样
```

### 6.2 nn.Module 和 nn.functional 结合使用

- 如果模型具有可学习的参数，最好用 `nn.Module`
- 否则既可以使用 `nn.functional`，也可以使用 `nn.Module` 

>  例如：激活函数、池化层没有可学习参数，可以使用对应的 `functional` 函数代替。而卷积层、线性层/全连接层需要学习参数，所以使用 `nn.Module` 
>
> 【推荐】dropout 虽然无参数学习，但推荐使用 `nn.Module`

例：混合使用

```python
# 混合使用
from torch.nn import functional as F


class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 不需要声明那些没有参数学习的层：池化等

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 池化直接写在前向传播里即可
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)  # 计算池化后的大小
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
x = torch.randn(64, 3, 32, 32)  # batch_size=64, channels=3, height=32, width=32
model = myNet()
out = model(x)
print(out.shape)  # torch.Size([64, 10])
```



## 7 优化器

PyTorch 将提供常用的优化方法，这些方法全部封装在 `torch.optim` 中

以 [1.2 多层感知机](#1.2 多层感知机) 为例，首先构建网络

```python
class MultiPerceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        # 新增参数：隐藏层神经元个数（形状）
        super().__init__()
        # 直接使用之前定义的线性层/全连接层 Linear
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)  # 使用激活函数，增加非线性因素（此处是逐个元素计算）
        y = self.layer2(x)
        return y
```

然后实例化网络

```python
# in_features=3, hidden_features=4, out_features=1
mlp = MultiPerceptron(3, 4, 1)
```

设置优化器和学习率（使用随机梯度下降优化器 SGD）

```python
# 设置优化器和学习率
from torch import optim

learning_rate = 0.9

# 为网络设置学习率，使用随机梯度下降优化器 SGD
optimizer = optim.SGD(params=mlp.parameters(), lr=learning_rate)  # 【重点】

# 下面就是网络的训练过程，这里我们只模仿更新一次
optimizer.zero_grad()  # 梯度清零，因为梯度累计效应

inputs = torch.randn(32, 3)  # batch_size=32, in_features=3
output = mlp(inputs)
output.backward(output)  # fake backward

optimizer.step()  # 执行优化
```

> 如果想为不同参数设置不同学习率

```python
# 为不同的参数分别设置不同的学习率
weight_params = [param for name, param in mlp.named_parameters() if name.endswith('.W')]
bias_params = [param for name, param in mlp.named_parameters() if name.endswith('.b')]

optimizer = optim.SGD([
    {'params': bias_params},
    {'params': weight_params, 'lr': 1e-2}
], lr=1e-5)
```



























