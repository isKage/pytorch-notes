# Pytorch 搭建神经网络（6）GPU 加速：CUDA 的使用

基于[《深度学习框架 Pytorch 入门与实践》陈云](https://book.douban.com/subject/27624483/) 

参考 [Github 的 pytorch-book 项目](https://github.com/chenyuntc/pytorch-book)

---

使用 GPU 加速技术，可以大幅减少训练时间。Pytorch 中的 `Tensor` 张量和 `nn.Module` 类就分为 CPU 和 GPU 两种版本。一般使用 `.cuda()` 和 `.to(device)` 方法实现从 CPU 迁移到 GPU ，从设备迁移到设备。

> 但 `Tensor` 和 `nn.Module` 使用 `.cuda()` 方法时返回的对象不同。
>
> - `Tensor.cuda()` 返回一个新对象，即拷贝了一份张量到 GPU ，之前的张量仍然储存在 CPU 
> - `nn.Module` 实例化后的 `module` 使用 `module.cuda()` 直接将所有数据推送到 GPU 不备份自己，即 `module.cuda()` 与 `module = module.cuda()` 等价

> `.to(device)` 可以更灵活地在不同设备上迁移



## 1 .cuda() 方法

### 1.1 张量 .cuda() 返回新的对象

```python
import torch

t = torch.tensor([1, 2, 3])

if torch.cuda.is_available():  # 检查 CUDA 是否可用
    t_cuda = t.cuda()  # 将张量 t 移动到 CUDA 设备
    print(t.is_cuda)
    print(t_cuda.is_cuda)
else:
    print("no CUDA")
```

```python
[Out]: False
	   True
```

> `.cuda()` 等价于 `.cuda(0)`  or `.cuda(device=0)` 迁移到第 0 块 GPU 上

### 1.2 module.cuda() 返回自己

```python
from torch import nn

module = nn.Linear(3, 4)

if torch.cuda.is_available():
    module.cuda(device=0)
    print(module.weight.is_cuda)  # True
else:
    print("no CUDA")
```

```python
[Out]: True
```

> 迁移到 GPU 本质都是对张量 Tensor 做变换，所以对于模型 module ，也是其权重等参数进行迁移



## 2 .to(device) 方法

`.to(device)` 方法可以指定设备

```python
# 使用.to方法，将 Tensor 转移至第 0 块GPU上
t = torch.tensor([1, 2, 3])

if torch.cuda.is_available():
    t_cuda = t.to('cuda:0')  # device: 'cuda:0'
    print(t_cuda.is_cuda)
else:
    print("no CUDA")
```

```python
[Out]: True
```



## 3 损失函数迁移到 GPU

大部分的损失函数都属于 `nn.Module` ，在使用 GPU 时，建议使用 `.cuda` 或 `.to` 迁移到 GPU 。

```python
if torch.cuda.is_available():
    # 以交叉熵损失函数为例
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 3]))

    # 张量 Tensor 迁移到 GPU
    inputs = torch.randn(4, 2).cuda()
    target = t.Tensor([1, 0, 0, 1]).long().cuda()

    # loss = criterion(input, target)  # 报错：计算损失函数的参数为迁移到 GPU
    
    # 正确：迁移损失函数
    criterion.cuda()
    loss = criterion(inputs, target)
    print(criterion._buffers)
```

```python
[Out]: OrderedDict([('weight', tensor([1., 3.], device='cuda:0'))])
```



## 4 torch.cuda.device() 指定默认设备

使用 `torch.cuda.device()` 指定默认设备，则不需要每次调用 `.cuda` 或 `.to` 方法。

```python
# 指定默认使用GPU "cuda:0"
with torch.cuda.device(0):    
    # 在 GPU 上构建Tensor
    a = torch.cuda.FloatTensor(2, 3)

    # 将 Tensor 转移至 GPU
    b = torch.FloatTensor(2, 3).cuda()
    
    print(a.get_divice)
    print(b.get_divice)

    c = a + b
    print(c.get_divice)
```

或者使用 `torch.set_default_tensor_type()` 方法，指定张量类型

```python
torch.set_default_tensor_type('torch.cuda.FloatTensor') # 指定默认 Tensor 的类型为GPU上的FloatTensor
a = torch.ones(2, 3)
print(a.is_cuda)  # True

torch.set_default_tensor_type('torch.FloatTensor') # 恢复默认
```



## 5 多 GPU 操作

多个 GPU 方便快捷地指定迁移设备。

### 5.1 方法一：调用 `torch.cuda.set_device()` 

例如：指定先调用 `torch.cuda.set_device(1)` 指定默认使用 GPU "cuda:1" ，后续的 `.cuda()` 都无需更改，切换 GPU 只需修改这一行代码



### 5.2 方法二：设置环境变量 `CUDA_VISIBLE_DEVICES`

例如当 `CUDA_VISIBLE_DEVICE=1` 时，代表优先使用 GPU "cuda:1" 而不是 GPU "cuda:0" 。此时调用`tensor.cuda()` 会将Tensor转移至 GPU "cuda:1"

`CUDA_VISIBLE_DEVICES` 还可以指定多个 GPU 。例如 `CUDA_VISIBLE_DEVICES=0,2,3` 代表按照 GPU "cuda:0", "cuda:2", "cuda:3" 的顺序使用 GPU 。此时 `cuda(0)` 迁移到 GPU "cuda:0" ，而 `.cuda(1)` 迁移到 GPU "cuda:2" ，`.cuda(2)` 迁移到 GPU "cuda:3" 。



### 5.3 设置 `CUDA_VISIBLE_DEVICES` 的方法：

- 法一：命令行中执行 `CUDA_VISIBLE_DEVICES=0,1 python main.py` 来运行主程序
- 法二：程序中编写 
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```
- 法三：IPython 或者 Jupyter notebook 中（即 `.ipynb` 文件），则可以使用魔法方法 
```python
%env CUDA_VISIBLE_DEVICES=0,1
```



## 6 CPU 与 GPU 并存

考虑到不同电脑可能会有差异，例如当遇到无 GPU 的主机时，容易出现不兼容。（例如：无法迁移到 GPU），所以建议判断是否存在 GPU ，再迁移数据。

```python
# 【推荐】如果用户具有 GPU 设备，那么使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

确定了设备之后，再迁移

```python
x = torch.randn(2, 3)
x_to = x.to(device)
print(x_to.device)

# device(type='cuda', index=0)
```



## 7 张量指定设备

### 7.1 创建张量时指定设备

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_cpu = torch.empty(2, device='cpu')
print(x_cpu, x_cpu.is_cuda)  # False

x_gpu = torch.empty(2, device=device)
print(x_gpu, x_gpu.is_cuda)  # True
```

### 7.2 new_* 保留原属性

```python
# new_*() : 保留原 Tensor 的设备属性

y_cpu = x_cpu.new_full((3, 4), 9)  # new_full : 用 9 填充形状 [3, 4] 的张量
print(y_cpu, y_cpu.is_cuda)  # False

y_gpu = x_gpu.new_zeros(3, 4)  # new_zeros : 填充 0
print(y_gpu, y_gpu.is_cuda)  # True
```

### 7.3 *_like 保留原属性

```python
# 使用ones_like或zeros_like可以创建与原Tensor大小类别均相同的新Tensor

z_cpu = torch.ones_like(x_cpu)  # 大小相同，设备相同
print(z_cpu, z_cpu.is_cuda)  # False


z_gpu = torch.zeros_like(x_gpu)  # 大小相同，设备相同
print(z_gpu, z_gpu.is_cuda)  # True
```







