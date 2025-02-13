# 搭建神经网络：深入学习 Pytorch

本库是关于学习 Pytorch
的笔记和相关实践代码，更好的阅读体验可以查看我的知乎专栏 [Pytorch 教程](https://zhuanlan.zhihu.com/column/c_1864780737208799232)
或者前往我的博客查看 [isKage\`Blog : 深度学习 Pytorch 完整教程](https://blog.iskage.online/categories/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-Pytorch-%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B/) 。

笔记以 `markdown` 格式编写，代码以 `.ipynb` 文件编写，内容均分章节存放，可查看相关目录，或者下载到本地自行运行。

---

> 创建虚拟环境并配置 Pytorch 可参见 [为搭建神经网络创建虚拟环境全流程](https://zhuanlan.zhihu.com/p/22230632892)
> 和 [Conda 创建虚拟环境全流程](https://zhuanlan.zhihu.com/p/21629604277) 。

## 1 下载到本地查看

在终端进入一个目录

```bash
cd <目录名>
```

然后执行

```bash
git clone https://github.com/isKage/pytorch-notes.git
```

即可下载，随后前往之前的目录 `<目录名>` 里查看是否成功下载文件夹 `pytorch-notes`

## 2 笔记目录

1. [张量 Tensor 与自动求导 Autograd](./01_tensor_autograd)
    1. [张量 Tensor](./01_tensor_autograd/note01_1_tensor.md)
    2. [自动求导 Autograd 和计算图](./01_tensor_autograd/note01_2_autograd.md)
2. [神经网络工具 nn 模块](./02_neural_network_tools)
    1. [Pytorch 的 nn 模块](./02_neural_network_tools/note02_nn.md)
3. [数据处理](./03_data_processing)
    1. [数据处理之 Dataset 和 Dataloader](./03_data_processing/note03_dataprocessing.md)
4. [可视化与 GPU 加速](./04_tensorboard_and_gpu)
    1. [Tensorboard 可视化工具](./04_tensorboard_and_gpu/note04_1_tensorboard.md)
    2. [GPU 加速: CUDA](./04_tensorboard_and_gpu/note04_2_cuda.md) 

## 3 友链

1. 代码和笔记主要基于 [《深度学习框架 Pytorch 入门与实践》陈云](https://book.douban.com/subject/27624483/)
   ，参考了公开库 [Github 的 pytorch-book 项目](https://github.com/chenyuntc/pytorch-book)
   和 [GitHub 的 pytorch-handbook 项目](https://github.com/zergtant/pytorch-handbook) 。

2. 关注我的知乎账号 [Zhuhu](https://www.zhihu.com/people/--55-97-8-41) 不错过我的笔记更新。

3. 我会在个人博客 [isKage\`Blog](https://blog.iskage.online/) 更新相关项目和学习资料。


