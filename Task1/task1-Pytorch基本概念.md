# <center>Pytorch 基本概念</center>

**Task1 任务：**
>1. 什么是Pytorch，为什么选择Pytroch？
2. Pytroch的安装
3. 配置Python环境
4. 准备Python管理器
5. 通过命令行安装PyTorch
6. PyTorch基础概念
7. 通用代码实现流程(实现一个深度学习的代码流程)

## 1. 什么是Pytorch，为什么选择Pytroch？
#### 1.1 Torch是什么？  
Torch是一个Numpy类似的张量（Tensor）操作库，与Numpy不同的是Torch对GPU支持的很好，Lua是Torch的上层包装。
>Lua是一个小巧的脚本语言。是巴西里约热内卢天主教大学里的一个研究小组，由Roberto Ierusalimschy、Waldemar Celes 和 Luiz Henrique de Figueiredo所组成并于1993年开发。 其设计目的是为了嵌入应用程序中，从而为应用程序提供灵活的扩展和定制功能。Lua由标准C编写而成，几乎在所有操作系统和平台上都可以编译，运行.

#### 1.2 Pytorch和Torch的关系！
PyTorch和Torch使用包含所有相同性能的C库：TH, THC, THNN, THCUNN，并且它们将继续共享这些库。

其实PyTorch和Torch都使用的是相同的底层，只是使用了不同的上层包装语言。

注：LUA虽然快，但是太小众了，所以才会有PyTorch的出现。

#### 1.3 Pytorch是什么？
PyTorch是一个基于Torch的Python开源机器学习库，用于自然语言处理等应用程序。 它主要由Facebook的人工智能研究小组开发。Uber的"Pyro"也是使用的这个库。

Pytorch是一个Python包，提供两个高级功能：
- 具有强大的GPU加速的张量计算(如Numpy)
- 包含自动求导系统的深度神经网络(呃...)

#### 1.4 Why Pytorch？
Pytorch和Tensorflow的对比，附上一个推荐的[链接](https://zhuanlan.zhihu.com/p/28636490)：  

PyTorch更有利于研究人员、爱好者、小规模项目等快速搞出原型。而TensorFlow更适合大规模部署，特别是需要跨平台和嵌入式部署时。

- PyTorch算是相当简洁优雅且高效快速的框架
- 设计追求最少的封装，尽量避免重复造轮子
- 算是所有的框架中面向对象设计的最优雅的一个，设计最符合人们的思维，它让用户尽可能地专注于实现自己的想法
- 大佬支持,与google的Tensorflow类似，FAIR的支持足以确保- PyTorch获得持续的开发更新
- 不错的的文档（相比FB的其他项目，PyTorch的文档简直算是完善了，参考Thrift），PyTorch作者亲自维护的论坛 供用户交流和求教问题
- 入门简单

## 2. Pytorch的安装
是在不使用anaconda的情况下安装的：

**使用pip安装pytorch及torchvision：**  
>pip install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl  
pip install torchvision

**torchvision库简介：**  
- torchvisoin是独立于Pytorch的关于图像操作的一些方便工具库；  
- torchvision的详细介绍：https://pypi.org/project/torchvision/  
- torchvision主要包括以下几个包：
>vison.datasets:几个常用视觉数据集，可以下载和加载
>vison.modles:流行的模型，例如AlexNet，VGG，ResNet和Densenet以及训练好的参数
>vison.utils:用于把形似3×H×W的张量保存到硬盘中，给一个mini-batch的图像可以产生一个图像网格

## 3. 配置Python环境

Win10 64位 + Python3.6 + Pytorch-1.1.0  ·o·


## 4. Pytorch基础概念

#### 4.1 [Tensor(张量)](https://github.com/zuiing/Pytorch/blob/master/Task1/Tensor(%E5%BC%A0%E9%87%8F)%E7%BB%83%E4%B9%A0.ipynb)
Tensors与Numpy中的 ndarrays类似，但是在PyTorch中 Tensors可以使用GPU进行计算。

Tensor是神经网络框架中重要的基础数据类型，可以简单理解为N维数组的容器对象。Tensor之间的通过运算进行连接，从而形成计算图。

#### 4.2 Autograd:自动求导
PyTorch 中所有神经网络的核心是 autograd 包。

autograd包为张量上的所有操作提供了自动求导。 它是一个在运行时定义的框架，这意味着反向传播是根据你的代码来确定如何运行，并且每次迭代可以是不同的。

#### 4.3 神经网络
torch.nn模块提供了创建神经网络的基础构件，这些层都继承自Module类。当实现神经网络时需要继承自此模块，并在初始化函数中创建网络需要包含的层，并实现forward函数完成前向计算，网络的反向计算会由自动求导机制处理。


更详细的Pytorch基础概念可参考另一篇[博客](https://blog.csdn.net/zzulp/article/details/80573331)。

## 5. [通用代码实现流程](https://github.com/zuiing/Pytorch/blob/master/Task1/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B-%E5%9B%9E%E5%BD%92.ipynb)
根据莫烦大神的视频，通过关系拟合-回归来说明通用代码实现的流程。

