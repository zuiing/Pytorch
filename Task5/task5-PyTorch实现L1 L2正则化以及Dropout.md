# <center>PyTorch实现L1，L2正则化以及Dropout</center>

>1. 了解知道Dropout原理
>2. 用代码实现正则化(L1、L2、Dropout）
>3. Dropout的numpy实现
>4. PyTorch中实现dropout
>5. 参考资料：PyTorch 中文文档



##  1. Dropout原理

#### 1.1 Drop简介

在机器学习的模型中，如果模型的参数太多，而训练样本又太少，训练出来的模型很容易产生过拟合的现象。在训练神经网络的时候经常会遇到过拟合的问题，过拟合具体表现在：模型在训练数据上损失函数较小，预测准确率较高；但是在测试数据上损失函数比较大，预测准确率较低。

过拟合是很多机器学习的通病。如果模型过拟合，那么得到的模型几乎不能用。为了解决过拟合问题，一般会采用模型集成的方法，即训练多个模型进行组合。此时，训练模型费时就成为一个很大的问题，不仅训练多个模型费时，测试多个模型也是很费时。

综上所述，训练深度神经网络的时候，总是会遇到两大缺点：

- 容易过拟合

- 费时

Dropout可以比较有效的缓解过拟合的发生，在一定程度上达到正则化的效果。

**在正向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征。**



##  2. 用代码实现正则化

#### 2.1 **L1，L2正则化**

机器学习中几乎都可以看到损失函数后面会添加一个额外项，常用的额外项一般有两种，一般L1正则化和L2正则化，或者L1范数和L2范数。L1正则化和L2正则化可以看做是损失函数的惩罚项。所谓『惩罚』是指对损失函数中的某些参数做一些限制。一般回归分析中回归w表示特征的系数，正则化则是对系数做了处理（限制)。

- L1正则化是指权值向量w中各个元素的**绝对值之和**，通常表示为||w||1
- L2正则化是指权值向量w中各个元素的**平方和再求平方根**，通常表示为||w||2

二者的作用：

- L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择
- L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合



'''

    import torch
    import torchvision
    
    model=torchvision.models.vgg16()
    
    def Regularization(model):
        L1=0
        L2=0
        for param in model.parameters():
            L1+=torch.sum(torch.abs(param))
            L2+=torch.norm(param,2)
        return L1,L2
    mnist=torchvision.datasets.mnist

'''

## 3. Dropout的numpy实现

'''

    import numpy as np
    
    class Dropout():
    
        def __init__(self, prob=0.5):
            self.prob = prob
            self.params = []
    
        def forward(self, X):
            self.mask = np.random.binomial(1, self.prob, size=X.shape) / self.prob
            out = X * self.mask
            return out.reshape(X.shape)
    
        def backward(self, dout):
            dX = dout * self.mask
            return dX, []

'''

## 4. [PyTorch中实现dropout](https://github.com/zuiing/Pytorch/blob/master/Task5/Dropout%E7%BC%93%E8%A7%A3%E8%BF%87%E6%8B%9F%E5%90%88(ptyorch).ipynb)

'''

    class Model(nn.Module):
        def __init__(self):
            super(Model,self).__init__()
            # 定义多层神经网络
            self.fc1 = torch.nn.Linear(8,6)
            self.fc2 = torch.nn.Linear(6,4)
            self.fc3 = torch.nn.Linear(4,1)
    
        def forward(self,x):
            x = F.relu(self.fc1(x))            # 8->6
            x = F.dropout(x,p=0.5)             #dropout 1 此处为dropout
            x = F.relu(self.fc2(x))            #-6->4
            x = F.dropout(x,p=0.5)             # dropout 2  #此处为drouout
            y_pred = torch.sigmoid(self.fc3(x))         # 4->1 ->sigmoid 
            # warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead."
            return y_pred
'''














