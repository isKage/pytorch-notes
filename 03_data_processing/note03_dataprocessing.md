# Pytorch搭建神经网络（4）数据处理 Dataset 和 Dataloader


基于[《深度学习框架 Pytorch 入门与实践》陈云](https://book.douban.com/subject/27624483/) 

参考 [Github 的 pytorch-book 项目](https://github.com/chenyuntc/pytorch-book)

---

本章主要讲解如何使用 Pytorch 实现深度学习/神经网络里的数据处理。相比于搭建已知的神经网络，对数据的处理更为复杂困难。【数据处理非常重要且困难！！】

Pytorch 针对数据处理，提供了两个重要的类 `Dataset` 和 `Dataloader`

## 1 Dataset 类

在 PyTorch 中，数据加载可以通过 `Dataset`类加载。如果要自定义的数据集，需要继承 `Dataset` 类，并且必须实现：

- `__getitem__()` ：返回一条数据，或一个样本。`obj[index]` 等价于 `obj.__getitem__(index)`
- `__len__()` ：返回样本量。`len(obj)` 等价于 `obj.__len__()`

### 1.1 准备数据

以 [Kaggle 经典案例 "Dogs vs. Cats"](https://www.kaggle.com/competitions/dogs-vs-cats) 的数据为例，构造数据集。有关数据下载，可以前往 [Kaggle 比赛官网下载](https://www.kaggle.com/competitions/dogs-vs-cats/data) 。但完整数据集太大，可以下载部分图片，或者[点击链接](https://cloud-iskage.oss-cn-shanghai.aliyuncs.com/packages/cat_dog.zip)下载我存储的小部分样本。或者去往来我的 GitHub 库中下载 [github.com/isKage/pytorch-notes](https://github.com/isKage/pytorch-notes)。

Kaggle 比赛 "Dogs vs. Cats" 是一个分类问题：判断一张图片是狗还是猫。在该问题中，所有图片都存放在一个文件夹下，可以根据文件名的前缀得到它们的标签值（ `dog` 或者 `cat` ）

```bash
tree data/cat_dog/
data/cat_dog/
├── cat.13.jpg
├── cat.14.jpg
├── cat.16.jpg
├── cat.18.jpg
├── dog.2.jpg
├── dog.3.jpg
├── dog.4.jpg
└── dog.5.jpg
```

### 1.2 自定义数据集

- 导入必要的库：`PIL` 库用于读取图片，`os` 库用于给出路径（以免不同电脑路径组合方式不同）
- 自定义数据集：继承 `torch.utils.data` 的 `Dataset` 类
- 编写 `__init__` 方法：先初始化图片路径，但暂时不读取图片
- 编写 `__getitem__` 方法：真正读取图片，并定义标签，并转为张量
- 编写 `__len()__` 方法：返回样本量

```python
import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import numpy as np

class DogCat(Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)  # 所有图片前的绝对路径表
        # 不实际加载图片，只指定路径，当调用 __getitem__ 时才读取图片，以节省内存
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # 标签设置：dog -> 1， cat -> 0
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        # 真正读取图片
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = torch.tensor(array)  # 转为张量
        return data, label

    def __len__(self):
        return len(self.imgs)  # 返回样本数
```

### 1.3 读取数据集

实例化自定义的数据集 `DogCat` ，因为 `__init__(self, root)` 初始化方法里需要参数图片存储的文件夹/路径 `root`

```python
# 读取数据集
dataset = DogCat('./data/cat_dog/')  # 图片存储在 ./data/cat_dog/ 文件夹内
```

### 1.4 读取数据集里的数据

使用方法 `__getitem__` 获取真实数据（此处为图片）`img` 和标签 `label` 

```python
# 获取第一个数据，包含标签和图片
img, label = dataset[0]  # 相当于 dataset.__getitem__(0)

print(img.shape)
# torch.Size([374, 500, 3]) 说明图片已转为张量

print(label)
# 0 说明是 cat
```

逐个批量读取

```python
for img, label in dataset:
    print("tensor's shape: {}, label: {}".format(img.shape, label))
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739264855512.png)

> 但是，我们注意到，图片大小形状不一致，且没有进行标准化，所以需要数据预处理



## 2 torchvision 库

Pytorch 的 `torchvision` 包提供了许多工具用来处理计算机视觉问题。其中 `torchvison.transforms` 可以方便快速地对图像进行操作。

### 2.1 torchvison.transforms 的常见操作

可以去往 [Pytorch 官网](https://pytorch.org/vision/stable/transforms.html) 查询具体使用。

#### 2.1.1 仅支持 PIL Image

- `RandomChoice` ：在一系列 transforms 操作中随机执行一个操作
- `RandomOrder` ：以随意顺序执行一系列 transforms 操作

#### 2.1.2 仅支持 Tensor

- `Normalize` ：标准化，即减去均值，除以标准差
- `RandomErasing` ：随机擦除 Tensor 中一个矩形区域的像素。
- `ConvertImageDtype` ：将 Tensor 转换为指定的类型，并进行相应的缩放

#### 2.1.3  PIL Image 与 Tensor 相互转换

- `ToTensor` ：将 $H \times W \times C$ 形状的 `PIL Image` 对象转换成形状为 $C \times H \times W$ 的 `Tensor` 。同时会自动将像素值从 [0, 255] 归一化至 [0, 1] （C 位通道数）
- `ToPILImage` ：将 `Tensor` 转为 `PIL Image` 对象

#### 2.1.4 既支持 PIL Image ，又支持 Tensor 

- `Resize` ：调整图片尺寸
- `CenterCrop` `RandomCrop` `RandomResizedCrop` `FiveCrop` ： 按照不同规则对图像进行裁剪
- `RandomAffine` ：随机进行仿射变换，保持图像中心不变
- `RandomGrayscale` ：随机将图像变为灰度图
- `RandomHorizontalFlip` `RandomVerticalFlip` `RandomRotation` ：随机水平翻转、垂直翻转、旋转图像

> 如果需要对图片进行多个操作，可以通过 `transforms.Compose` 将这些操作拼接起来。

> 注意，这些操作定义后以对象的形式存在，真正使用时需要调用 `__call__` 方法。例如，要将图片的大小调整至 $10 \times 10$ ，首先应构建操作 `trans = Resize((10, 10))` ，然后调用 `trans(img)` 



### 2.2 transforms 操作

#### 2.2.1 定义变化操作序列

```python
from torchvision import transforms

# 定义变换操作
transform = transforms.Compose([
    transforms.Resize(224),  # 缩放图片 (PIL Image), 保持长宽比不变, 使最短边缩放到 224 像素
    transforms.CenterCrop(224),  # 从图片中间切出 224x224 的图片
    transforms.ToTensor(),  # 将图片 (PIL Image) 转成 Tensor , 自动归一化至 [0, 1]
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至 [-1, 1] , 规定均值和标准差 , 因为图片为 3 维
])
```

#### 2.2.2 自定义数据集中加入变化与否参数

将有关变换序列参数加入初始化方法里，同时在 `__getitem()__` 里加入变换

```python
# 增加参数 transform 传入变化序列
class DogCat(Dataset):
    def __init__(self, root, transform=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transform = transform  # 初始化 transforms 操作

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # 标签设置：dog -> 1， cat -> 0
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        # 真正读取图片
        data = Image.open(img_path)
        if self.transform:
            data = self.transform(data)  # 直接进行 transform 变化
        return data, label

    def __len__(self):
        return len(self.imgs)  # 返回样本数
```

#### 2.2.3 实例化数据集

传入变化序列 `transform=transform`

```python
# 读取数据集
dataset = DogCat('./data/cat_dog/', transform=transform)  # 使用 transform 进行变换
```

#### 2.2.4 展示结果

可以发现数据格式统一成了我们需要的样子

```python
for img, label in dataset:
    print(img.shape, label)
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739276735119.png)

> 注意：自定义数据集需要考虑到当 `transform = None` 时的情形。在上述定义的数据集里，当默认`transform = None` 时不会报错，此时 `img` 是 `PIL Image` 对象。

```python
no_transform = DogCat('./data/cat_dog/')
img, label = no_transform[0]
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739276821066.png)

### 2.3 torchvision 封装的常见数据集

`torchvision` 封装了常用的 dataset ，例如：`CIFAR-10`、`ImageNet`、`COCO`、`MNIST`、`LSUN` 等数据集。可以通过诸如 `torchvision.datasets.CIFAR10` 的命令进行调用，具体使用方法参考[官方文档](https://pytorch.org/vision/stable/datasets.html)。

#### 2.3.1 ImageFolder 结构

常见数据集结构 `ImageFolder` ：数据集假设所有的图片按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名。即需要目录结构为：【可将上文的 cat_dog 数据集自主修改为符合 ImageFolder 要求的目录结构】

```bash
tree data/cat_dog_imagefolder/
data/cat_dog_imagefolder/
├── cat
│   ├── cat.13.jpg
│   ├── cat.14.jpg
│   ├── cat.16.jpg
│   └── cat.18.jpg
└── dog
    ├── dog.2.jpg
    ├── dog.3.jpg
    ├── dog.4.jpg
    └── dog.5.jpg
```

- `ImageFolder` 类的参数

```python
ImageFolder(root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None)

- root
root 路径目录下存放着不同类别的文件夹

- transform
对 PIL Image 进行相关操作，transform 的输入是使用 loader 读取图片的返回对象 (一般是 PIL Image)

- target_transform
对 label 的操作

- loader
指定加载图片的函数，默认操作是读取为 PIL Image 对象

- is_valid_file
获取图像路径，检查文件的有效性
```

> `target_transform` 对标签的操作默认，则会返回一个字典。形如 `{文件夹1名: 0, 文件夹2名: 1, 文件夹3名: 2, ...}` 的字典，每个类的文件夹名和数字对应。可以通过 `.class_to_idx` 查看对应关系。

#### 2.3.2 ImageFolder 创建数据集 Dataset

- 实例化数据集（未指定变换，直接读取 PIL Image）

```python
from torchvision.datasets import ImageFolder
dataset = ImageFolder('./data/cat_dog_imagefolder')
```

获取一下标签 `label` 对应关系

```python
dataset.class_to_idx

# {'cat': 0, 'dog': 1}
```

于是我们知道：`0` 对应 `'cat'` ，`1` 对应 `'dog'`

- 查看数据集：返回图片路径和对应的标签

```python
dataset.imgs
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739278461256.png)

- 查看具体数据（图片）：因为没有进行操作，默认读取的是 PIL Image 对象

`dataset[0]` 获取数据集第 0 个样本，包括 `img` 和 `label` ，其中第 0 个是 `img` ，第 1 个是 `label`

```python
dataset.__getitem__(0)[0]  # 等价于 dataset[0][0], 获取 img
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739279152983.png)

- 获取标签，返回类别名

```python
# 辅助返回类别名，不重要
def get_class_by_idx(idx):
    for key_class, val_idx in dataset.class_to_idx.items():
        if val_idx == idx:
            return key_class
    return 'Not found'


# 获取标签 dataset[0][1]
print("The label is: {} meaning {}".format(
    dataset.__getitem__(0)[1],
    get_class_by_idx(dataset[0][1])
))

''' [Out]: The label is: 0 meaning cat '''
```

- 【加入变换】：一般情况下，我们会加入变换 transforms

```python
# 加入变换
transform = transforms.Compose([
    transforms.Resize(224),  # 缩放图片 (PIL Image), 保持长宽比不变, 使最短边缩放到 224 像素
    transforms.CenterCrop(224),  # 从图片中间切出 224x224 的图片
    transforms.ToTensor(),  # 将图片 (PIL Image) 转成 Tensor , 自动归一化至 [0, 1]
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至 [-1, 1] , 规定均值和标准差 , 因为图片为 3 维
])

# 重新构造数据集
dataset = ImageFolder('./data/cat_dog_imagefolder', transform=transform)
```

结果：查看图片张量形状，因为神经网络中一般图像的通道数在第一个维度，而 PIL Image 类型通道数在第三个维度，所以变换十分必要。

```python
dataset.__getitem__(0)[0].shape  # or dataset[0][0].shape

''' [Out]: torch.Size([3, 224, 224]) '''
```

也正是因为 Tensor 和 PIL Image 对通道数位置要求的不同，二者转换往往需要多一步：

```python
to_img = transforms.ToPILImage()
to_img(dataset[0][0] * 0.5 + 0.5)  # 因为变换指定了均值和方差后进行了归一化，所以要返归一化
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739279681882.png)



## 2 DataLoader 类

在训练神经网络时，一次处理的对象是一个 `batch` 的数据，同时还需要对一批数据进行打乱顺序和并行加速等操作。为此，PyTorch提供了 `DataLoader` 实现这些功能。

**参数**

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

- `dataset` : 数据集 Dataset 类
- `batch_size=1` :  批量 batch 的大小
- `shuffle=False` : 是否打乱数据顺序
- `sampler=None` : 用于从数据集中抽取样本的采样器，可以自定义采样策略
- `batch_sampler=None` : 定义如何批量抽取样本
- `num_workers=0` : 用多进程加载的进程数，0 代表不使用多进程
- `collate_fn=None` : 多样本拼接 batch 的方法，一般默认
- `pin_memory=False` : 是否将数据保存在 pin memory 区，pin memory 中的数据转移到 GPU 速度更快
- `drop_last=False` : Dataset 中的数据不一定被 batch_size 整除时，若 drop_last 为 True ，则将多出来的数据丢弃
- `timeout=0` : 进程读取数据的最大时间，若超时则丢弃数据
- `worker_init_fn=None` : 每个 worker 的初始化函数（num_workers=0 则无影响）
- `prefetch_factor=2` : 每个 worker 预先加载的样本数

### 2.1 Dataloader 的使用

```python
# dataloader
from torch.utils.data import DataLoader

dataset = DogCat('./data/cat_dog/', transform=transform)

dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)  # 加载 dataset, 批量大小为 3, 打乱顺序, 单进程, 不丢弃最后数据
```

- 每个 batch 数据形如 `torch.Size([3, 3, 224, 224])` ：第一个 3 代表批量 batch 大小，第二个 3 代表图片通道数，最后代表 224x224 的图片大小

```python
for img_batch, label_batch in dataloader:
    print(img_batch.shape, label_batch.shape)
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739329118058.png)

> 最后一个批量为 2 是因为图片总共 8 张，而批量按 3 个数据一取，剩余 2 个数据

- 安装迭代器的方式取数据：dataloader 可以通过迭代器的方式取数据 `iter(Dataloader)`

```python
dataiter = iter(dataloader)
imgs, labels = next(dataiter)
print(imgs.shape)  # torch.Size([3, 3, 224, 224])
```

> 迭代器 `iter()`

```python
# 迭代器补充
a = [1, 2, 3]
b = iter(a)
print(next(b)) # 1
print(next(b)) # 2
```

### 2.2 自定义 collate_fn 避免读取失败

若某个样本无法读取，此时利用 Dataset 类时， `__getitem__`函数中会抛出异常。

- 例如：可手动创建一个新的文件夹，添加一个空文件命名为 `dog.fail.jpg`

```bash
tree data/cat_dog_fail/
data/cat_dog_fail/
├── cat.13.jpg
├── cat.14.jpg
├── cat.16.jpg
├── cat.18.jpg
├── dog.2.jpg
├── dog.3.jpg
├── dog.4.jpg
├── dog.5.jpg
└── dog.fail.jpg
```

此时如果仍然利用之前自定义的 Dataset 类 `DogCat` ，则会在后面真正读取数据时报错

```python
# 报错 : UnidentifiedImageError: cannot identify image file
# for img, label in dataset:
#     print(img.shape, label)
```

#### 2.2.1 解决方法一：返回 None

当读取失败时，返回 None ，然后自定义 `collate_fn` 让 Dataloader 加载时跳过

定义新的 Dataset 类，继承之前自定义的 `DogCat` ，尝试使用父类的 `__getitem__()` ，如果失败则返回 `(None, None)` 代表 `(数据, 标签)`

```python
class NewDogCat(DogCat):  # 继承之前自定义的 Dataset 类 DogCat
    # 修改 __getitem__() 方法
    def __getitem__(self, index):
        try:
            # 调用父类读取图片的方法 __getitem__() 等价于 DogCat.__getitem__(self, index)
            return super().__getitem__(index)
        except:
            # 数据=None, 标签=None
            return None, None
```

自定义 `collate_fn` 解决 Dataloader 读取数据异常：collate_fn 的传入参数 batch 是一个列表，形如 `[(data1, label1), (data2, label2), ...]` 删去里面为空的元祖，然后采用 Dataloader 默认的拼接方式返回最终的批量数据

```python
from torch.utils.data.dataloader import default_collate  # 导入 Dataloader 默认的拼接方式

# 定义 collate_fn 函数，删去 None 值
def delete_fail_sample(batch):
    # batch 是一个 list，每个元素是 dataset 的返回值，形如 (data, label)
    batch = [_ for _ in batch if _[0] is not None]  # 过滤为 None 的数据
    if len(batch) == 0: 
        return torch.Tensor()  # 如果整个数据集都是空的
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据
```

开始读取数据：实例化新的 dataset

```python
dataset = NewDogCat('data/cat_dog_fail/', transform=transform)
dataset[5] # (None, None) 第 5 个读取了错误的图片
```

Dataloader 读取数据：批量为 2 ，使用自定义 collate_fn 函数，单进程，打乱顺序，不丢弃最后数据

```python
dataloader = DataLoader(dataset, 2, collate_fn=delete_fail_sample, num_workers=0, shuffle=True, drop_last=False)
for img_batch, label_batch in dataloader:
    print(img_batch.shape, label_batch.shape)
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739330925980.png)

> 此处有 2 个批量不统一：一个是由于有一个图片没有读取成功；另一个是因为总共 9 张图片，无法整除 2

#### 2.2.2 解决方法二：随机读取正常数据 【推荐】

不再返回 `(None, None)` 而是随机返回其他正常数据，这样可以避免因读取失败带来的形状不统一

```python
import random

# 随机读取数据
class NewDogCat(DogCat):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except:
            new_index = random.randint(0, len(self) - 1)  # 随机返回一张正常数据
            return self[new_index]
```

Dataloader 读取数据时不再需要自定义 collate_fn

```python
dataset = NewDogCat('data/cat_dog_fail/', transform=transform)
dataloader = DataLoader(dataset, 2, collate_fn=None, num_workers=0, shuffle=True, drop_last=False)
for img_batch, label_batch in dataloader:
    print(img_batch.shape, label_batch.shape)
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739331306790.png)

> 此处唯一一个不统一形状是由于 9 不能被 2 整除，可以通过丢弃实现 `drop_last=True`）

### 2.3 随机采样

当 `DataLoader` 的 `shuffle=True` 时，会自动调用采样器 `RandomSampler` 打乱数据。默认的采样器是`SequentialSampler`，它会按顺序一个一个进行采样。除此之外，`WeightedRandomSampler` 也是一个很常见的采样器，它会根据每个样本的权重选取数据。

**参数**

- `weights` ：每个样本的权重，权重越大的样本被选中的概率越大

- `num_samples` ：选取的样本总数，一般小于总数据量（可以大于）

- `replacement` （可选）：默认/建议为 True，即允许重复采样同一个数据。当为 True 时，此时就算总样本数不足 `num_samples` 也会依靠重复取样达到

**使用案例**

- 首先读取数据集，并设置权重列表

```python
dataset = DogCat('data/cat_dog/', transform=transform)  # 总共 8 张图

# 设置权重：假设 cat 的图片被取出的概率是 dog 的概率的 2 倍
weights = [2 if label == 0 else 1 for data, label in dataset]
# 两类图片被取出的概率与 weights 的绝对大小无关，只和比值有关

print(weights)
# [2, 1, 1, 1, 1, 2, 2, 2] -> cat (label=0) 设权重为 2 ; dog (label=1) 设权重为 1
```

- 设定取样样本总数为 9（大于总图片数 8）选择 `replacement=True` 允许重复取样

```python
from torch.utils.data.sampler import WeightedRandomSampler

sampler = WeightedRandomSampler(weights=weights, num_samples=9, replacement=True)

dataloader = DataLoader(dataset, batch_size=3, sampler=sampler)
```

- 结果：`cat (label=0)` 的数量大约为 `dog (label)` 的 2 倍

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1739333471748.png)

> 【注意】注意次数样本数为 9 已经大于总图片数 8，说明当允许重复取样 `replacement=True` 时可以自行补充样本。同时原来 Dataloader 中的 `shuffle` 操作也会失效，因为随机取样完全由取样器决定

