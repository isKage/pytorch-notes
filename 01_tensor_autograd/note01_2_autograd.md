# Pytorch 搭建神经网络（2）自动求导autograd、反向传播backward与计算图

基于[《深度学习框架 Pytorch 入门与实践》陈云](https://book.douban.com/subject/27624483/) 

参考 [Github 的 pytorch-book 项目](https://github.com/chenyuntc/pytorch-book)

参考 [GitHub 的 pytorch-handbook 项目](https://github.com/zergtant/pytorch-handbook)

---

`torch.autograd` 提供了一套自动求导方式，它能够根据前向传播过程自动构建计算图，执行反向传播。

## 1 autograd 的数学原理：计算图

计算图原理可以查看 **cs231n** 课程讲解：【计算图的原理非常重要！】或者见[后文分析](#3)

英文官网 [https://cs231n.github.io/](https://cs231n.github.io/)

b站 课程整理 [BV1nJ411z7fe](https://www.bilibili.com/video/BV1nJ411z7fe?p=8&spm_id_from=333.788.videopod.episodes) 【反向传播章节】

b站 中文讲解 [【子豪兄】精讲CS231N斯坦福计算机视觉公开课（2020最新）](https://www.bilibili.com/video/BV1K7411W7So?p=4&vd_source=67ce2d561f3b6dc9d7cff375959101a2)



## 2 autograd 的使用：requires_grad & backward

### 2.1 requires_grad 属性

只需要对Tensor增加一个 `requires_grad=True` 属性，Pytorch就会自动计算 `requires_grad=True` 属性的 Tensor，并保留计算图，从而快速实现反向传播。

```python
# Method 1
x = torch.randn(2, 3, requires_grad=True)

# Method 2
x = torch.rand(2, 3).requires_grad_()

# Method 3
x = torch.randn(3, 4)
x.requires_grad = True

print(x.requires_grad)  # True
```

### 2.2 backward 反向传播

反向传播函数的使用：其中第一个参数 `tensors` 传入用于计算梯度的张量，格式和各个参数

```python
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False)
```

- `tensors`：用于计算梯度的Tensor，如`torch.autograd.backward(y)`，等价于`y.backward()`。

- `grad_tensors`：形状与tensors一致，对于`y.backward(grad_tensors)`，grad_tensors相当于链式法则${\mathrm{d}z \over \mathrm{d}x}={\mathrm{d}z \over \mathrm{d}y} \times {\mathrm{d}y \over \mathrm{d}x}$中的${\mathrm{d}z} \over {\mathrm{d}y}$。【结合例子理解见后】
- `retain_graph`：计算计算图里每一个导数值时需要保留各个变量的值，retain_graph 为 True 时会保存。【结合例子理解见后】

#### 2.2.1 requires_grad 属性的传递

- 例：`a` 需要求导，`b` 不需要，`c` 定义为 `a + b` 的元素加和

```python
a = torch.randn(2, 3, requires_grad=True)
b = torch.zeros(2, 3)
c = (a + b).sum()  # c 受 a 的影响，c.requires_grad = True

a.requires_grad, b.requires_grad, c.requires_grad
# (True, False, True)
```

#### 2.2.2 is_leaf 叶子结点

对于计算图中的Tensor而言， `is_leaf=True` 的Tensor称为Leaf Tensor，也就是计算图中的叶子节点。

- `requires_grad=False` 时，无需求导，故为叶子结点。
- 即使 `requires_grad=True` 但是由用户创建的时，此时它位于计算图的头部（叶子结点），它的梯度会被保留下来。

```python
# 仍然是上面的例子
a.is_leaf, b.is_leaf, c.is_leaf
# (True, True, False)
```

### 2.3 autograd 利用计算图计算导数

利用 autograd 计算导数，对于函数 $y=x^2e^x$，它的导函数解析式为
$$
\begin{equation}
\dfrac{d\ y}{d\ x} = 2xe^x + x^2e^x
\end{equation}
$$
定义计算 y 函数和计算解析式导数结果函数

```python
# autograd 求导
# y = x^2 * e^x
def f(x):
    y = x * x * torch.exp(x)
    return y


def df(x):
    df = 2 * x * torch.exp(x) + x * x * torch.exp(x)
    return df
```

- 例：随机赋值

```python
x = torch.randn(2, 3, requires_grad=True)
y = f(x)

# y = 
# tensor([[0.1387, 0.4465, 0.4825],
#         [0.1576, 4.1902, 0.5185]], grad_fn=<MulBackward0>)
```

```python
y.backward(gradient=torch.ones(y.size()))  # 指定 dy/dx = dy/dx * 1 的 dy/dx
# torch.autograd.backward(y, grad_tensors=torch.ones(y.size()))  # 或者
```

```python
print(x.grad)  # 反向传播后才能取到 y 关于 x 的导数（已经代入了此时 x 的值）
# tensor([[-0.4497,  2.1766, -0.2087],
#         [-0.4567, 11.4700, -0.1244]])

print(df(x))   # 解析求出的导数值
# tensor([[-0.4497,  2.1766, -0.2087],
#         [-0.4567, 11.4700, -0.1244]], grad_fn=<AddBackward0>)
```

`x.grad & df(x)` 二者是在数值上是一样的



## 3 反向传播与计算图<a id='3'></a>

### 3.1 计算图原理：链式法则

根据链式法则

$dz/dy = 1,\ dz/db = 1$

$dy/dw = x,\ dy/dx = w$

$dz/dx = dz/dy \times dy/dx = 1 \times w,\ dz/dw = dz/dy \times dy/dw = 1 \times x$

只要存储结点的导数和值便可通过简单的乘法计算所有导数

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739002830328.png)

按照上图构造

```python
# 计算图
x = torch.ones(1)
b = torch.rand(1, requires_grad = True)
w = torch.rand(1, requires_grad = True)
y = w * x # 等价于 y = w.mul(x)
z = y + b # 等价于 z = y.add(b)

x.requires_grad, b.requires_grad, w.requires_grad, y.requires_grad, z.requires_grad
# (False, True, True, True, True)
```

### 3.2 grad_fn 查看反向传播函数

`grad_fn` 可以查看这个结点的函数类型

```python
z.grad_fn  # <AddBackward0 at 0x7f96b951ba90>  Add 加法，因为 z = y + b
y.grad_fn  # <MulBackward0 at 0x7f96b951b400>  Mul 乘法，因为 y = w * x

w.grad_fn, x.grad_fn, b.grad_fn # (None, None, None) 叶子结点是 grad_fn=None
```

`grad_fn.next_functions` 获取 grad_fn 的输入，返回上一步的反向传播函数

```python
z.grad_fn.next_functions  # z 前是 y 和 b
# ((<MulBackward0 at 0x7f96b951b400>, 0),  # y = w * x 是 mul
#  (<AccumulateGrad at 0x7f96b95c6af0>, 0))  # b 是叶子结点，需要求导 AccumulateGrad

y.grad_fn.next_functions  # y 前是 w 和 x
# ((<AccumulateGrad at 0x7f9678466160>, 0),  # w 是叶子结点，需要求导 AccumulateGrad
#  (None, 0)  # x 是叶子节点，x.requires_grad=False 不需要求导 None
```

### 3.3 retain_graph 的使用（仅叶子结点）

如果不指定 `retain_graph=True` ，则在反向传播后，会自动清除变量值。

例如：计算 `w.grad` w 的梯度时，需要 x 的值 （$dy/dw = x$）

> 注意：x.requires_grad=False 不需要求导，故 `x.grad` 报错

```python
z.backward(retain_graph=True)
print(w.grad)
# tensor([1.])  # 确实是我们之前设的 x = torch.ones(1) 相匹配
```

```python
# 再次运行，梯度累加
z.backward()
print(w.grad)
# tensor([1.])  # 1 + 1 = 2 累加，所以之前 grad_fn 取名为 AccumulateGrad
```

### 3.4 关闭反向传播

某一个节点 `requires_grad `被设置为 `True` ，那么所有依赖它的节点 `requires_grad` 都是 `True`。有时不需要对所有结点都反向传播（求导），从而来节省内存。

```python
x = torch.ones(1)
w = torch.rand(1, requires_grad=True)
y = x * w

x.requires_grad, w.requires_grad, y.requires_grad  # y.requires_grad = True
# (False, True, True)
```

下面我们来关闭关于 `y` 的反向传播

- 法一：`with torch.no_grad():`

```python
with torch.no_grad():
    x = torch.ones(1)
    w = torch.rand(1, requires_grad=True)
    y = x * w
    
x.requires_grad, w.requires_grad, y.requires_grad  # y.requires_grad = False
# (False, True, False)
```

- 法二：设置默认 `torch.set_grad_enabled(False)`

```python
torch.set_grad_enabled(False) # 更改默认设置

x = torch.ones(1)
w = torch.rand(1, requires_grad = True)
y = x * w

x.requires_grad, w.requires_grad, y.requires_grad  # y.requires_grad = False
# (False, True, False)

torch.set_grad_enabled(True) # 恢复默认设置
```

### 3.5 `.data` 从计算图取出Tensor的值

修改张量的数值，又不影响计算图，使用 `tensor.data` 方法

```python
x = torch.ones(1, requires_grad = True)
x_clone = x.data

x.requires_grad, x_clone.requires_grad  # x_clone 独立于原来的计算图
# (True, False)
```

### 3.6 存储非叶子结点的梯度

在计算图流程中，非叶子结点求导后其导数值便立刻被清除。可以使用 `autograd.grad` 或 `hook` 方法保留

```python
# autograd.grad & hook
x = torch.ones(1, requires_grad = True)
w = torch.ones(1, requires_grad = True)
y = w * x  # 非叶子结点
z = y.sum()  # 非叶子结点
```

```python
z.backward()
x.grad, w.grad, y.grad  # 非叶子结点 y.grad = None
# (tensor([1.]), tensor([1.]), None)
```

> 若为叶子结点可以采用 `z.backward(retain_graph=True)` 的方式

- 法一：`torch.autograd.grad()`

```python
# 使用 torch.autograd.grad() 直接取梯度
x = torch.ones(1, requires_grad = True)
w = torch.ones(1, requires_grad = True)
y = x * w
z = y.sum()

torch.autograd.grad(z, y)  # z.backward() 并直接取 y.grad()
# (tensor([1.]),)
```

- 法二：`hook`

标准格式

```python
# hook是一个函数，输入是梯度，不应该有返回值
def variable_hook(grad):
    print('y.grad：', grad)

x = torch.ones(1, requires_grad = True)
w = torch.ones(1, requires_grad = True)

y = x * w
# 注册hook
hook_handle = y.register_hook(variable_hook)

z = y.sum()
z.backward()

# 除非每次都要使用 hook，否则用完之后记得移除 hook
hook_handle.remove()

# y.grad： tensor([1.])
```



## 4 案例：线性回归

```python
import torch
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

def get_fake_data(batch_size=16):
    # 产生随机数据：y = 2 * x + 3，加上噪声
    x = torch.rand(batch_size, 1) * 5  # 扩大一些，以免噪声太明显
    y = x * 2 + 3 + torch.randn(batch_size, 1)
    return x, y

# 设置随机数种子，保证结果可复现
torch.manual_seed(1000)

x, y = get_fake_data()

# plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
# plt.show()

# 初始化
w = torch.rand(1, 1, requires_grad=True)  # w.shape = torch.Size([1, 1]) 因为 [8, 1] * [1, 1] -> [batch_size, 1] 和 y 维度相同
b = torch.zeros(1, 1, requires_grad=True)

losses = np.zeros(200)  # 存储损失值
lr = 0.005  # 学习率
EPOCHS = 200  # 迭代次数

for epoch in range(EPOCHS):
    x, y = get_fake_data(batch_size=32)

    # 前向传播 计算损失
    y_pred = x.mm(w) + b.expand_as(y)  # expand_as(y) 是广播机制，即将 b 复制成和 y 相同性质的张量 [1, 1] -> [batch_size, 1]
    loss = 0.5 * (y_pred - y) ** 2  # MSE 均方误差，这是对张量 y 逐元素计算
    loss = loss.sum()  # 累和成一个数
    losses[epoch] = loss.item()

    # 反向传播
    loss.backward()

    ''' 取 .data 是因为每一轮是根据随机生成的 batch_size 个点训练，但我们希望存储的是全局参数 w, b '''
    ''' 故每次依据样本点更新全局参数，而不是改批次的参数 '''
    # 更新参数
    w.data.sub_(lr * w.grad.data)  # 或者 w.data = w.data - lr * w.grad.data
    b.data.sub_(lr * b.grad.data)

    # 梯度清零
    w.grad.data.zero_()  # 不清零，梯度会不断累加
    b.grad.data.zero_()

    if epoch % 10 == 0:  # 每隔 10 次扔出当前训练情况
        print("Epoch: {} / {}, Parameters: w is {}, b is {}, Loss: {}".format(epoch, EPOCHS, w.item(), b.item(), losses[epoch]))

print("Epoch: {} / {}, Parameters: w is {}, b is {}, Loss: {}".format(EPOCHS, EPOCHS, w.item(), b.item(), losses[-1]))
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739010576001.png)

- GPU 加速

```python
import torch
import numpy as np
from matplotlib import pyplot as plt

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_fake_data(batch_size=16):
    # 产生随机数据：y = 2 * x + 3，加上噪声
    x = torch.rand(batch_size, 1, device=device) * 5  # 将数据移动到 GPU
    y = x * 2 + 3 + torch.randn(batch_size, 1, device=device)  # 将数据移动到 GPU
    return x, y

# 设置随机数种子，保证结果可复现
torch.manual_seed(1000)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1000)  # 为 CUDA 设置随机种子

# 初始化参数，并将参数移动到 GPU
w = torch.rand(1, 1, requires_grad=True, device=device)  # 将 w 移动到 GPU
b = torch.zeros(1, 1, requires_grad=True, device=device)  # 将 b 移动到 GPU

losses = np.zeros(200)  # 存储损失值
lr = 0.005  # 学习率
EPOCHS = 200  # 迭代次数

for epoch in range(EPOCHS):
    x, y = get_fake_data(batch_size=32)

    # 前向传播 计算损失
    y_pred = x.mm(w) + b.expand_as(y)  # expand_as(y) 是广播机制，即将 b 复制成和 y 相同性质的张量 [1, 1] -> [batch_size, 1]
    loss = 0.5 * (y_pred - y) ** 2  # MSE 均方误差，这是对张量 y 逐元素计算
    loss = loss.sum()  # 累和成一个数
    losses[epoch] = loss.item()

    # 反向传播
    loss.backward()

    # 更新参数
    w.data.sub_(lr * w.grad.data)  # 或者 w.data = w.data - lr * w.grad.data
    b.data.sub_(lr * b.grad.data)

    # 梯度清零
    w.grad.data.zero_()  # 不清零，梯度会不断累加
    b.grad.data.zero_()

    if epoch % 10 == 0:  # 每隔 10 次打印当前训练情况
        print("Epoch: {} / {}, Parameters: w is {}, b is {}, Loss: {}".format(epoch, EPOCHS, w.item(), b.item(), losses[epoch]))

print("Epoch: {} / {}, Parameters: w is {}, b is {}, Loss: {}".format(EPOCHS, EPOCHS, w.item(), b.item(), losses[-1]))
```





























































