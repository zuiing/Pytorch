## 【Task6(2天)】PyTorch理解更多神经网络优化方法

>1. 了解不同优化器
>2. 书写优化器代码
>3. Momentum
>4. 二维优化，随机梯度下降法进行优化实现
>5. Ada自适应梯度调节法
>6. RMSProp
>7. Adam
>8. PyTorch种优化器选择



### 1. 了解不同的优化器

 #### 1.1 优化器基类：Optimizer

当数据、模型和损失函数确定，任务的数学模型就已经确定，接着就是选择一个合适的优化器对该模型进行优化。

下面将介绍Pytorch中优化器的方法和使用方法：

**优化器基类（Optimizer）**

​		PyTorch中所有的优化器（如：optim.Adadelta optim.SGD optim.RMSprop等）均是Optimizer的子类，Optimizer中定义了一些常用的方法，有zero_grad()、step(closure)、state_dict()、load_state_dict(state_dict)和add_param_group(param_group)...

- **参数组(param_group)的概念**

  optimizer对参数的管理是基于组的概念，可以为每一组参数配置特定的lr,momentum,weight_decay等等。

  参数组在optimizer中表现为一个list(self.param_groups)，其中每个元素是dict，表示一个参数及其相应配置，在dict中包含'params'、'wright_decay'、'lr'、'momentum'等字段。

- **zero_grad()**

  功能：将梯度清零。

  有PyTorch不会自动清零梯度，所以在每一次更新前都会进行此操作。

- **state_dict()**

  功能：获取模型当前的参数，以一个有序字典形式返回。

  这个有序字典中，key是各层参数名，value是参数。

- **load_param_group()**

  功能：给optimizer管理的参数组中增加一组参数，可为该组参数定制lr，momanutm，weight_decay等，在finetune中常使用。

- **step(closure)**

  功能：执行一步权值更新，其中可传入参数closure(一个闭包)。

#### 1.2 PyTorhch的优化器

PyTorch提供了十个优化器，常用的有SGD、ASGD、Rprop、RMSprop、Adam等。

**梯度下降**

常用三种变形：BGD（批量梯度下降），SGD（随机梯度下降），MBGD（小批量梯度下降），三种形式对应不同数据量的情况。

- **BGD（Batch gradient descent，批量梯度下降）**

  优点：由全部数据集确定的方向能够更好的代表样本总体，更准确地向极值方向下降，若目标函数为凸函数，BGD一定能够得到全局最优解。非凸函数也可以得到局部最小值。可以实现为并行计算

  缺点：BGD是在一次更新中，对整个数据集计算梯度，在数据量规模不大时，计算速度慢的问题还不明显，但是随着数据量的增大，其速度慢的弊端会非常棘手，并且，计算过程中并不能及时地放入新的数据，以实时更新模型

- **SGD（Stochastic Gradient Descent，随机梯度下降）**

  梯度更新规则：每次迭代仅使用一个样本对参数进行更新，迭代速度快，如果数据量非常大，那么SGD有可能只需要遍历其中部分数据，就能得到最优解。相较于BGD，每次计算都是喂进去全部数据，一次迭代计算量大，也不能快速得到最优解；SGD虽然存在一定的随机性，造成准确度有所下降，但是从期望的角度来看，其价值对于代价是可接受的
  优点：避免了每次迭代对全部数据的遍历，计算速度和参数迭代速度明显加快
缺点：SGD因为更新非常频繁，会造成cost function剧烈震荡；可能最终收敛到局部最优；难以实现并行计算

- **MBGD（Mini-Batch Gradient Descent，小批量梯度下降）**

  梯度更新规则：本方法是对BGD和SGD两种方法的折中，即，每次迭代，使用batch_size个数据进行计算和更新参数
  优点：对于“小批量”的batch_size，使用矩阵形式计算（可以并行化），不会比SGD每次迭代一个样本慢很多，但是使用batch_size个数据进行迭代，又可以大大提高计算的速度，同时准确度相比SGD也提高了许多。
  缺点：batch_size的选择比较考究，在合理范围内，batch_size增大可以有效的提高效率，但是盲目增大batch_size会对内存造成过大的压力，花费时间会变大。所以batch_size的选择比较重要。batch_size一般取值50~256

参考《[快速上手笔记，PyTorch模型训练实用教程](https://mp.weixin.qq.com/s/c7QEnZ0_NTY1aUaoZ4nT7g)》

## 2. [书写优化器代码]()



参考[莫烦视频](<https://morvanzhou.github.io/tutorials/machine-learning/torch/3-06-optimizer/>)