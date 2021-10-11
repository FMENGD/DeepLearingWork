# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:39:51 2020

@author: FMENG
"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils #第一部分，初始化
import reg_utils #第二部分，正则化
import gc_utils #第三部分，梯度校验

####实现正则化
##读取并绘制数据集
train_X,train_Y,test_X,test_Y = reg_utils.load_2D_dataset(is_plot=False)
"""
每一个点代表球落下的可能的位置，蓝色代表己方的球员会抢到球，红色代表对手的球员会抢到球，我们要做的就是使用模型来画出一条线，来找到适合我方球员能抢到球的位置

我们需要做以下三件事，来对比不同的模型的优劣：
1、不使用正则化
2、使用正则化
    2.1 使用L2正则化
    2.2 使用随即节点删除
"""

def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,is_plot=True,lambd=0,keep_prob=1):
    """
    实现一个三层的神经网络：Linear -> Relu -> Linear -> Relu -> Linear -> Sigmoid
    
    参数：
        X - 输入的数据，维度为（2.要训练/测试的数量）
        Y - 标签，【0（蓝色）|1（红色）】，维度为（1，对应的是输入数据的标签）
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每10000次打印一次，但是每1000次记录一个成本值
        is_plot - 是否绘制梯度下降的曲线图
        lambd - 正则化的超参数，实数
        keep_prob - 随机删除节点的概率
    返回：
        parameters - 学习后的参数
    """
    
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],20,3,1]
    
    #初始化参数
    parameters = reg_utils.initialize_parameters(layers_dims)
    
    #开始学习
    for i in range(0,num_iterations):
        #前向传播
        ##时候随机删除节点
        if keep_prob == 1:
            ##不随机删除节点
            a3,cache = reg_utils.forward_propagation(X,parameters)
        elif keep_prob < 1:
            ##随机删除节点
            a3,cache = forward_propagation_with_dropout(X,parameters,keep_prob)
        else:
            print("keep_prob参数错误！程序退出")
            exit
        
        ###计算成本
        ##是否使用二范数
        if lambd == 0:
            ##不使用L2正则化
            cost = reg_utils.compute_cost(a3,Y)
        else:
            ###使用L2正则化
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)
            
        ##反向传播
        ##可以同时使用L2正则化和随机删除节点，但本次实验不同时使用
        
        assert(lambd == 0 or keep_prob == 1)
        
        ##两个参数的使用情况
        if (lambd == 0 and keep_prob == 1):
            #不使用L2正则化和不使用随机删除节点
            grads = reg_utils.backward_propagation(X,Y,cache)
        elif lambd != 0:
            ####使用正则化，不使用随机删除节点
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob < 1:
            ###使用随机删除节点，不使用L2正则化
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)
        
        ###更新参数
        parameters = reg_utils.update_parameters(parameters,grads,learning_rate)
        
        ##记录并打印成本
        if i % 1000 == 0:
            ##记录成本
            costs.append(cost)
            if(print_cost and i % 10000 == 0):
                #打印成本
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))
    
    
    #是否绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(x1,10000)')
        plt.title('Learning rate = '+ str(learning_rate))
        
        
    #返回学习后的参数
    return parameters

####不适用正则化下模型的效果
parameters = model(train_X,train_Y,is_plot=True)
print("训练集：")
predictions_train = reg_utils.predict(train_X,train_Y,parameters)
print("测试集：")
predictions_test = reg_utils.predict(test_X,test_Y,parameters)


##将数据的分割曲线画出来
plt.title("Model without regularzation")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.40])
reg_utils.plot_decision_boundary(lambda x:reg_utils.predict_dec(parameters,x.T),train_X,train_Y)
    
###在不使用正则化时，分割线又明显的过拟合性 所以接下来使用L2正则化

########L2 正则化
"""
为了避免过度拟合采用L2正则化，由原来的成本函数(1)变为现在的函数(2)：
    J = -1/m∑(y(i)log(a[L](i)) +(1-y(i))log(1-a[L](i)))  .....(1)
    
    J(正则化) = -1/m ∑（y(i)log(a[L](i)) + (1-y(i))log(1-a[L](i))）+ 1/m * λ/2ΣΣΣ Wk,j[l]平方   ....(2)
"""
#开始实现相关函数
def compute_cost_with_regularization(A3,Y,parameters,lambd):
    """
    实现公式2的L2正则化计算成本
    
    参数：
        A3 - 正向传播输出的结果，维度为（输出节点的数量，训练/测试的数量）
        Y - 标签向量，与数据一一对应，维度为（输出节点的数量，训练/测试的数量）
        parameters - 包含模型学习后的参数字典
        
    返回：
        cost - 使用公式2计算出来的正则化损失的值
    
    """
    
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = reg_utils.compute_cost(A3,Y) #原来的交叉熵损失
    
    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m) #加入的正则化项损失值
        
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

##因为改变了成本函数，我们也必须改变向后传播函数，所有的梯度都必须根据这个信的成本值来计算
    
def backward_propagation_with_regularization(X,Y,cache,lambd):
    """
    实现我们添加了L2正则化的模型的后向传播
    
    参数：
        X - 输入数据集，维度为（输入节点数量，数据集里面的数量）
        Y - 标签，维度为（输出节点的数量，数据集里面的数量）
        cache - 来自forward_propagation()的cache输出  ##cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
        lambd - regularization超参数，实数
        
    返回：
        gradients - 一个包含了每个参数、激活值和预激活值变量的梯度字典
    """
    
    m = X.shape[1]
    
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache
    
    dZ3 = A3 - Y
    
    dW3 = (1/m) * np.dot(dZ3,A2.T) + ((lambd * W3) / m)
    db3 = (1/m) * np.sum(dZ3,axis=1,keepdims=True)
    
    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    dW2 = (1/m) * np.dot(dZ2,A1.T) + ((lambd * W2) / m)
    db2 = (1/m) * np.sum(dZ2,axis=1,keepdims=True)
    
    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = (1/m) * np.dot(dZ1,X.T) + ((lambd * W1) / m)
    db1 = (1/m) * np.sum(dZ1,axis=1,keepdims=True)
    
    gradients = {
            "dZ3":dZ3,"dW3":dW3,"db3":db3,"dA2":dA2,
            "dZ2":dZ2,"dW2":dW2,"db2":db2,"dA1":dA1,
            "dZ1":dZ1,"dW1":dW1,"db1":db1
            }
    
    return gradients

##放到模型中跑一下
parameters = model(train_X,train_Y,lambd=0.7,is_plot=True)
print("使用正则化，训练集：")
predictions_train = reg_utils.predict(train_X,train_Y,parameters)
print("使用正则化，测试集：")
predictions_test = reg_utils.predict(test_X,test_Y,parameters)

###看一下分类结果
plt.title("Model with L2-regularizations")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x:reg_utils.predict_dec(parameters,x.T),train_X,train_Y)
    
"""
总结：
     λ的值是可以使用开发集调整时的超参数.  L2正则化会使决策边界更加平滑。
     通过削减成本函数中权重的平方值，可以将所有权重值逐渐改变到到较小的值.
     L2正则化对以下内容有影响：
     成本计算       ： 正则化的计算需要添加到成本函数中
     反向传播功能     ：在权重矩阵方面，梯度计算时也要依据正则化来做出相应的计算
     重量变小（“重量衰减”) ：权重被逐渐改变到较小的值。
"""

#随机删除节点
"""
最后，我们使用Dropout来进行正则化，Dropout的原理就是每次迭代过程中随机将其中的一些节点失效。
当我们关闭一些节点时，我们实际上修改了我们的模型。
丢弃的节点都不参与迭代时的前向和后向传播

总共四步：
1、在视频中，吴恩达老师讲解了使用np.random.rand() 来初始化和a[1]具有相同维度的d[1]。
在这里，我们将使用向量化实现，我们先来实现一个和A[1]具有相同的随机矩阵D[1].
2、如果D[1]低于keep_prob的值我们就把它设置为0，如果高于keep_prob的值，我们就把它设置为1
3、把A[1]更新为A[1] * D[1].
4、使用A[1]除以keep_prob。我们通过缩放计算成本的时候仍具有相同的期望值。

"""
    
def forward_propagation_with_dropout(X,parameters,keep_prob=0.5):
    """
    实现具有随机舍弃节点的前向传播：
    Linear -> Relu + dropout -> Linear -> Relu + dropout -> Linear -> sigmoid
    
    参数：
        X - 输入数据集。维度为（2，示例数）
        parameters - 包含参数“W1”,"b1","W2","b2","W3","b3"的python字典
            W1  - 权重矩阵，维度为（20，2） ##layers_dims=[2,20,3,1]
            b1 - 偏向量，维度为（20，1）
            W2 - 权重矩阵，维度为（3，20）
            b2 - 偏向量，维度为（3，1）
            W3 - 权重矩阵，维度为（1，3）
            b3 - 偏向量，维度为（1，1）
        keep_prob - 随机删除的概率，实数
    返回：
        A3 - 最后的激活值，维度为（1，1），正向传播的输出
        cache - 存储了一些用于计算反向传播的数值的元组
    """
    np.random.seed(1)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    #Linear -> Relu -> Linear -> Relu -> Linear -> sigmoid
    Z1 = np.dot(W1,X) + b1
    A1 = reg_utils.relu(Z1)
    
    #下面的步骤1-4对应于上述步骤的1-4
    D1 = np.random.rand(A1.shape[0],A1.shape[1])   #步骤1：初始化矩阵D1
    D1 = D1 < keep_prob                            #步骤2：将D1的值转换为0或者1
    A1 = A1 * D1                                   #步骤3：舍弃A1的一些节点（将它的值变为0或者False）
    A1 = A1 / keep_prob                            #步骤4：缩放未舍弃的节点（不为0）的值
    
    Z2 = np.dot(W2,A1) + b2
    A2 = reg_utils.relu(Z2)

    #下面的步骤1-4对应于上述步骤1-4
    D2 = np.random.rand(A2.shape[0],A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob
    
    Z3 = np.dot(W3,A2)+b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3)

    return A3,cache    
    
#改变了前向传播的算法，我们也需要改变后向传播。使用存储在缓存中的掩码D[1]和D[2]将舍弃的节点位置信息添加到第一个和第二个隐藏层

def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    """
    实现我们随机删除的模型的后向传播。
    
    参数：
        X - 输入数据集，维度为（2，示例数）
        Y - 标签，维度为（输出节点数量，示例数量）
        cache - 来自forward_propagation_with_dropout()的cache输出
        keep_prob - 随机删除的概率，实数
        
    返回：
        gradients - 一个关于每个参数、激活值、和预激活变量的梯度值字典
    """
    m = X.shape[1]
    (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3) = cache
    
    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3,A2.T)
    db3 = 1. / m * np.sum(dZ3,axis=1,keepdims=True)
    dA2 = np.dot(W3.T,dZ3)
    
    dA2 = dA2 * D2           #步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（任何数乘以0都为0）
    dA2 = dA2 / keep_prob    #步骤2：缩放未舍弃的节点（不为0）的值
    
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2,A1.T)
    db2 = 1. / m * np.sum(dZ2,axis=1,keepdims=True)   
    dA1 = np.dot(W2.T,dZ2)
    
    dA1 = dA1 * D1           #步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（任何数乘以0都为0）
    dA1 = dA1 / keep_prob    #步骤2：缩放未舍弃的节点（不为0）的值
    
    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1,X.T)
    db1 = 1. / m * np.sum(dZ1,axis=1,keepdims=True)
    
    gradients = {
            "dZ3":dZ3,"dW3":dW3,"db3":db3,"dA2":dA2,
            "dZ2":dZ2,"dW2":dW2,"db2":db2,"dA1":dA1,
            "dZ1":dZ1,"dW1":dW1,"db1":db1
            }
    
    return gradients
"""
我们前向和后向传播的函数都写好了，现在用dropout运行模型（keep_prob = 0.86）跑一波。
这意味着在每次迭代中，程序都可以24％的概率关闭第1层和第2层的每个神经元。
"""  
parameters = model(train_X,train_Y,keep_prob=0.86,learning_rate=0.3,is_plot=True)

print("使用随机删除节点，训练集：")
predictions_train = reg_utils.predict(train_X,train_Y,parameters)
print("使用随机删除节点，测试集：")
predictions_test = reg_utils.predict(test_X,test_Y,parameters)


##查看分类情况
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x : reg_utils.predict_dec(parameters,x.T),train_X,train_Y)  
    
##由上述我们看到，正则化会把训练集的准确度降低，但是测试集的准确度提高了。


   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    