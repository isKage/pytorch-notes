# Pytorch 搭建神经网络（5）可视化工具：TensorBoard

基于[《深度学习框架 Pytorch 入门与实践》陈云](https://book.douban.com/subject/27624483/) 

参考 [Github 的 pytorch-book 项目](https://github.com/chenyuntc/pytorch-book)

---

为了更直观地、实时地观察训练过程，使用一些可视化工具实现训练过程的图形化表达，以便直观地展现结果。

TensorBoard ：只要用户保存的数据遵循相应的格式，TensorBoard 就能读取这些数据，进行可视化。

## 1 下载 TensorBoard

最新版本的 Pytorch 在下载时已经配置了 TensorBoard 无需特别下载。如果没有下载 TensorBoard 包需要先在终端中输入
```bash
pip install tensorboard
```

## 2 创建 logger 对象

```python
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import models
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter  # 导入 tensorboard
```

```python
# 构建 logger 对象，指定 log 文件的保存路径 log_dir='logs'
logger = SummaryWriter(log_dir='logs')
```

- 此时已经可以通过执行：`tensorboard --logdir=path` 来访问可视化界面。

- `path` 填入 logger 对象文件保存路径 例如此处的 `'./logs'` 或 `'log'`

```bash
# 在终端输入以查看结果
tensorboard --logdir=logs
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739386419718.png)

- 打开浏览器，输入 `http://localhost:6006/` 查看

> 此时没有加入任何数据和图像，故界面如下


![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739386180385.png)

> 或者在 notebook 中查看

```python
%load_ext tensorboard
%tensorboard --logdir='logs'
```

## 3 添加数据 & 绘制图像

```python
# 使用 add_scalar 记录标量
for n_iter in range(100):
    logger.add_scalar(tag='Loss/train', scalar_value=np.random.random(), global_step=n_iter)
    logger.add_scalar(tag='Loss/test', scalar_value=np.random.random(), global_step=n_iter)
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739416635199.png)

> 如果不中断 `tensorboard --logdir=logs` 则可以通过刷新网页来查看。或者可以关闭 (`CTRL + C`) 后重新启动。

- 结束后，删除之前的文件

```bash
# Clear any logs from previous runs
rm -rf ./logs/
```

## 4 下载数据集 MNIST

以 `MNIST` 手写识别体数据集为例，首先先下载。可以直接调用 `torchvision.datasets.MNIST`

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # MNIST 是灰度图，单通道
])

# './data/' 指定下载的路径，download=True 表示下载，train=False 表示下载测试集，transform 指定变换
dataset = datasets.MNIST('./data/', download=True, train=False, transform=transform)
```

如果无法下载可以前往官网手动下载：

训练集：[train-images](https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz), [train-labels](https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz)

测试集：[t10k-images](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz), [t10k-labels](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz)

然后放入目录 `MNIST` 下的 `raw` 文件夹：

```bash
data
└── MNIST
    └── raw
        ├── t10k-images-idx3-ubyte.gz
        ├── t10k-labels-idx1-ubyte.gz
        ├── train-images-idx3-ubyte.gz
        └── train-labels-idx1-ubyte.gz
```

然后再运行之前的下载代码，将 `download=False`

```python
dataset = datasets.MNIST('./data/', download=False, train=False, transform=transform)
```

```bash
data
└── MNIST
    └── raw
        ├── t10k-images-idx3-ubyte
        ├── t10k-images-idx3-ubyte.gz
        ├── t10k-labels-idx1-ubyte
        ├── t10k-labels-idx1-ubyte.gz
        ├── train-images-idx3-ubyte
        ├── train-images-idx3-ubyte.gz
        ├── train-labels-idx1-ubyte
        └── train-labels-idx1-ubyte.gz
```

## 5 常用函数

### 5.1 add_scalar 添加标量

```python
add_scalar(self, tag, scalar_value, global_step=None)
```

- `tag` ：标题名
- `scalar_value` ：标量数值
- `global_step` ：迭代批次

例如：从 [0, 99] 随机产生标量绘图

```python
# 使用 add_scalar 记录标量
for n_iter in range(100):
    logger.add_scalar(tag='Loss/train', scalar_value=np.random.random(), global_step=n_iter)
    logger.add_scalar(tag='Loss/test', scalar_value=np.random.random(), global_step=n_iter)
```

### 5.2 add_image 显示图像

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # MNIST 是灰度图，单通道
])

# './data/' 指定下载的路径，download=True 表示下载，train=False 表示下载测试集，transform 指定变换
dataset = datasets.MNIST('./data/', download=True, train=False, transform=transform)
dataloader = DataLoader(dataset, shuffle=True, batch_size=16)
```

```python
images, labels = next(iter(dataloader))
grid = torchvision.utils.make_grid(images)
```

```python
# 使用 add_image 显示图像
logger.add_image('images', grid, 0)

%load_ext tensorboard
%tensorboard --logdir='logs'
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739418831074.png)

### 5.3 add_graph 显示网络结构

```python
class myModel(nn.Module):
    def __init__(self, input_size=28, hidden_size=500, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = myModel()
logger.add_graph(model, images)  # 代入具体的数值计算 torch.Size([16, 1, 28, 28])
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739420352341.png)

### 5.4 add_histogram 显示直方图

```python
logger.add_histogram('normal', np.random.normal(0, 5, 1000), global_step=1)
logger.add_histogram('normal', np.random.normal(1, 2, 1000), global_step=10)
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739420726898.png)

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739420601056.png)

### 5.5 add_embedding 可视化

```python
# 使用 add_embedding 进行 embedding 可视化
dataset = datasets.MNIST('./data/', download=True, train=False)  # PIL Image 对象可以可视化
images = dataset.data[:100].float()  # 提取前 100 张图像并转为浮点数类型
label = dataset.targets[:100]  # 提取前 100 张图像的标签
features = images.view(100, 28 * 28)  # 将图像展平为 100 x 784 的矩阵
logger.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))  # 将嵌入数据记录到 TensorBoard
```



