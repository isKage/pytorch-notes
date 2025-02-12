# Pytorch 搭建神经网络（1）Tensor 张量数据结构

基于[《深度学习框架 Pytorch 入门与实践》陈云](https://book.douban.com/subject/27624483/) 

参考 [Github 的 pytorch-book 项目](https://github.com/chenyuntc/pytorch-book)

参考 [GitHub 的 pytorch-handbook 项目](https://github.com/zergtant/pytorch-handbook)

参考 [DeepSeek](https://www.deepseek.com/) 整理补充

---

首先，检查 `Pytorch` 是否安装

```python
import torch
print(torch.__version__)
>>> 2.2.2
```



`Tensor` 是可以理解为一个类似 `Numpy` 中的高维数组。

## 1 创建

- `torch.Tensor()` 分配空间

生成维度2x3的张量，并未赋值

```python
x = torch.Tensor(2, 3)
print(x)

# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
```

- `torch.tensor()` 需要具体的值进行创建

输入具体的值，直接生成

```python
y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)  # dtype 指定类型
print(y)

# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
```

- `torch.rand()` 使用正态分布随机初始化

```python
z = torch.rand(2, 3)
print(z)

# tensor([[0.1587, 0.9499, 0.1939],
#         [0.9741, 0.9309, 0.7463]])
```

## 2 查看形状

- 调用方法 `.shape` 或 `.size()`查看张量形状/维度

```python
print(x.shape)
# torch.Size([2, 3])
```

- 产看具体某个维度数（例如列数）

```python
print(x.size()[1])  # 3
print(x.size(1))   	# 3
print(x.shape[1])  	# 3
```

## 3 加法

- 使用`+` 作加法

```python
x = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float)  # 或者 torch.ones(2, 3)
y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
print(x + y)

# tensor([[2., 3., 4.],
#         [5., 6., 7.]])
```

> 这种加法不改变 `x, y` 的值

- 使用 `torch.add(x, y)` 作加法

```python
z = torch.Tensor(2, 3)  # 先分配好一个空间，不赋值
torch.add(x, y, out=z)
print(z)

# tensor([[2., 3., 4.],
#         [5., 6., 7.]])
```

> 这种加法不改变 `x, y` 的值

- 调用方法 `.add()` 和 `.add_()`

```python
print(y.add(x))  # y 不变
print(y)

print(y.add_(x)) # y 变为 x + y
print(y)
```

> 增加了 `_` 的方法会进行替换操作

## 4 索引

Tensor 的索引操作与 NumPy 类似

```python
# 索引
x = torch.rand(2, 3)
print(x)
# tensor([[0.3479, 0.8074, 0.2170],
#         [0.3419, 0.9281, 0.1364]])

print(x[:, 1])  # tensor([0.8074, 0.9281]) 			# 行全取，列取第一列（从0计数）
print(x[1, :])  # tensor([0.3419, 0.9281, 0.1364])  # 列全取，行取第一行（从0计数）
```

## 5 和 `Numpy` 的转换

- `torch.Tensor -> numpy.ndarray`

使用 `.numpy()` 方法从Tensor变为numpy.ndarray

```python
# numpy 相互转换
x = torch.ones(1, 3)
y = x.numpy()

print(x)		# tensor([[1., 1., 1.]])
print(type(x))	# <class 'torch.Tensor'>
print(y)		# [[1. 1. 1.]]
print(type(y))  # <class 'numpy.ndarray'>
```

- `numpy.ndarray -> torch.Tensor`

使用 `torch.from_numpy()` 函数从numpy.ndarray变为Tensor

```python
import numpy as np
y = np.ones((1, 3))
x = torch.from_numpy(y)
print(y)        # [[1. 1. 1.]]
print(type(y))  # <class 'numpy.ndarray'>
print(x)		# tensor([[1., 1., 1.]], dtype=torch.float64)
print(type(x))  # <class 'torch.Tensor'>
```

- 共享内存，通过上面方式转换后，`x, y` 是共享内存的，tenor改变，numpy.ndarray也改变

```python
print(x)  # tensor([[1., 1., 1.]])
print(y)  # [[1. 1. 1.]]
temp = torch.rand(1, 3)
x.add_(temp)
print(x)  # tensor([[1.5567, 1.5514, 1.0607]])
print(y)  # [[1.5567319 1.5514015 1.0607271]]
```

## 6 零维度张量/标量

Tensor 数据类型中维度为 `0` 称为标量（注意，虽然维度为0，但仍然不是 `int` 或是 `float` 这些一般 Python 数据类型）

```python
scaler = torch.tensor(9)
print(scaler)          # tensor(9)
print(scaler.shape)	   # torch.Size([])
```

如果想获得一般 Python 数据类型，可以使用方法 `.item()`

```python
print(scaler.item())   	     # 9
print(type(scaler.item()))   # <class 'torch.Tensor'> 变为 int
```

> 注意区分 0 维度标量和 1 维度张量
>
> 但是针对 1 维度张量，也可以使用 `.item()` 方法 

```python
vector = torch.tensor([9])
scaler = torch.tensor(9)
print(vector) 		 # tensor([9])
print(vector.shape)  # torch.Size([1])
print(scaler)        # tensor(9)
print(scaler.shape)  # torch.Size([])

# 针对 1 维度张量，也可以使用 `.item()` 方法 
print(vector.item()) # 9
print(type(vector.item()))  # <class 'torch.Tensor'> 变为 int
```

## 7 张量间的复制 `.detach()` `.clone()`

- `.clone()` 不共享内存，二者互不影响

```python
old = torch.tensor([[1, 2, 3], [4, 5, 6]])
new = old.clone()  # 不共享内存，二者互不影响
new[0] = 233

print(old)
# tensor([[1, 2, 3],
#          [4, 5, 6]]

print(new)
# tensor([[233, 233, 233],
#         [  4,   5,   6]])
```

- `.detach()` 共享内存，一者变则全变

```python
old = torch.tensor([[1, 2, 3], [4, 5, 6]])
new = old.detach() # 共享内存，一者变则全变
new[0] = 233

print(old)
# tensor([[233, 233, 233],
#         [  4,   5,   6]])

print(new)
# tensor([[233, 233, 233],
#         [  4,   5,   6]])
```

## 8 维度转变

PyTorch提供了许多维度变换方式：`view, reshape, permute, transpose`

### 8.1 维度交换 `permute` ` transpose`

使用 `permute` ` transpose` 对张量维度进行交换，例如维度为 2x3x4x5 ，希望变为 3x2x5x4，可以

```python
previous = torch.randn(2, 3, 4, 5)  # .randn 标准正态

# permute
new1 = previous.permute((1, 0, 3, 2))  # 填入一个元组，元组里的每个数字对应原来张量的维度序号，新的维度为(第1维度,第0维度,第3维度,第2维度)

# transpose
new2 = previous.transpose(0, 1)  # 第0维度和第1维度交换
new2 = new2.transpose(2, 3)  # 第2维度和第3维度交换

print(new1.shape)  # torch.Size([3, 2, 5, 4])
print(new2.shape)  # torch.Size([3, 2, 5, 4])
```

### 8.2 维度变换 `view` `reshape`

- `reshape` 无要求，可直接使用
- `view`只能用于内存中连续存储的 Tensor

```python
previous = torch.randn(2, 3, 4, 5)

# 注意总维度要正确：例如 2x3x4x5 = 120 = 6x20
new1 = previous.reshape(-1, 10)  # -1 表示自动计算维度，reshape 和 view 均可使用
new2 = previous.view(6, 20)

print(new1.shape)  # torch.Size([12, 10])
print(new2.shape)  # torch.Size([6, 20])
```

> 如果经过了 `permute` ` transpose` 维度交换，则需要先连续化内存，使用 `.contiguous()`

```python
previous = torch.randn(2, 3, 4, 5)

# 内存不连续
new = previous.permute((1, 0, 3, 2))  # torch.Size([3, 2, 5, 4])

# 连续化然后变换维度
current = new.contiguous().view(6, 20)
# current = new.reshape(6, 20)  # 或者直接 reshape

print(current.shape)  # torch.Size([6, 20])
```

### 8.3 维度压缩、扩展、拼接 `squeeze` `unsqueeze` `cat`

- 维度压缩 `torch.squeeze()`

`torch.squeeze(input, dim=None)` 用于移除张量中大小为1的维度。如果不指定 `dim`，则会移除所有大小为1的维度；如果指定了 `dim`，则只会移除该维度（如果该维度大小为1）

```python
# 压缩
x = torch.randn(1, 3, 1, 2)  # torch.Size([1, 3, 1, 2])

# 移除所有大小为1的维度
y = torch.squeeze(x)
print(y.shape)  			 # torch.Size([3, 2])

# 只移除第2个维度（索引从0开始）
z = torch.squeeze(x, dim=2)
print(z.shape) 				 # torch.Size([1, 3, 2])
```

- 维度扩展 `torch.unsqueeze()`

`torch.unsqueeze(input, dim)` 用于在指定的位置插入一个大小为1的维度。`dim` 参数指定了新维度插入的位置

```python
x = torch.randn(3, 2)    	# torch.Size([3, 2])

# 在第0维插入一个大小为1的维度
y = torch.unsqueeze(x, dim=0)
print(y.shape)           	# torch.Size([1, 3, 2])

# 在第1维插入一个大小为1的维度
z = torch.unsqueeze(x, dim=1)
print(z.shape)          	 # torch.Size([3, 1, 2])
```

- 维度拼接 `torch.cat()`

`torch.cat(tensors, dim=d)` 用于在指定的维度上拼接多个张量。所有张量在除了 `dim=d` 维度之外的其它维度上**必须具有相同的形状**

```python
x = torch.randn(1, 4)  	     # dim=1 维度均为 3 故可以在dim=0上拼接
y = torch.randn(2, 4)

# 在第0维上拼接
z = torch.cat((x, y), dim=0)
print(z.shape)  	         # torch.Size([3, 4])  # 1 + 2 = 3
```



## 9 GPU 加速

利用 GPU 的并行计算能力能加速模型的计算。Pytorch提供了2种将tensor推至GPU的方法。

- `.cuda()` 方法

```python
x = x.cuda()  # 将 Tensor 转移到默认的 GPU
```

- `.to(device)` 方法【推荐】

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = x.to(device)  # 将 Tensor 转移到指定的设备，如果失败则继续在CPU计算
```

```python
>>> import torch
>>> print(torch.cuda.is_available)
<function is_available at 0x0000024B63F50720>

>>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
>>> device
device(type='cuda', index=0)

>>> x = torch.randn(2, 3)
>>> x.to(device)
tensor([[-0.1888,  0.0827, -1.2929],
        [ 2.1295,  1.6174, -1.4917]], device='cuda:0')
```

































