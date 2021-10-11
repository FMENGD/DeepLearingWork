# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:37:35 2020

@author: FMENG
"""

#要实现的事情：
"""
1、初始化参数：
    1.1 使用0来初始化参数
    1.2 使用随机数来初始化参数
    1.3 使用抑梯度异常初始化参数
2、正则化模型：
    2.1 使用二范数对二分类模型正则化，尝试避免过拟合
    2.2 使用随机删除节点的方法精简模型。同样是为了避免过拟合
3、梯度校验：
    对模型使用梯度校验，检测它是否存在梯度下降的过程中出现误差过大的情况
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils #第一部分，初始化
import reg_utils #第二部分，正则化
import gc_utils #第三部分，梯度校验

plt.rcParams['figure.figsize'] = (7.0,4.0) #设置plots的初始值
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X,train_Y,test_X,test_Y = init_utils.load_dataset(is_plot=True)

"""
三种初始化方法：
·初始化为0：在输入参数中全部初始化为0，参数名为initialization = “zeros”
·初始化为随机数：把输入参数设置为随机值，权重初始化为大的随机值。参数名为initialization = “random”
·抑梯度异常初始化：参见梯度消失和梯度爆炸的那一个视频，参数名为initialization = “he”
"""

def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="he",is_plot=True):
    """
    实现一个三层的神经网络：Linear -> Relu -> Linear -> Relu -> Linear -> sigmoid
    
    参数：
        X - 输入的数据，维度为（2，要训练/测试的数量）
        Y - 标签，【0|1】，维度为（1，对应的是输入的数据的标签）
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代1000次打印一次
        initialization - 字符串类型，初始化的类型【“zeros”|"random"|"he"】
        is_plot - 是否绘制梯度下降的曲线图
    返回：
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]
    
    #选择初始化参数的类型
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    else:
        print("错误的初始化参数！程序推出")
        exit
    
    #开始学习
    for i in range(0,num_iterations):
        ##前向传播
        a3,cache = init_utils.forward_propagation(X,parameters)
        
        #计算成本
        cost = init_utils.compute_loss(a3,Y)
        
        ##反向传播
        grads = init_utils.backward_propagation(X,Y,cache)
        
        #更新参数
        parameters = init_utils.update_parameters(parameters,grads,learning_rate)
        
        ##记录成本
        if i % 1000 == 0:
            costs.append(cost)
            #打印成本
            if print_cost:
                print("第"+str(i)+"次迭代，成本值为：" + str(cost))
                
    #学习完毕，绘制成本曲线
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("learning rate =" + str(learning_rate))
        
    #返回学习完毕后的参数
    
    return parameters

#上述为实现的模型 下面尝试三种初始化
def initialize_parameters_zeros(layers_dims):
    """
    将模型的参数全部设置为0
    
    参数：
        layers_dims - 列表，模型的层数和对应的每一层的节点数量
    返回：
        parameters - 包含了所有W和b的字典
            W1 - 权重矩阵，维度为（layers_dims[1],layers_dims[0]）
            b1 - 偏置向量，维度为（layers_dims[1],1）
            ...
            WL - 权重矩阵，维度为（layers_dims[L],layers_dims[L-1]）
            bL - 偏置向量，维度为（layers_dims[L],1）
    """
    parameters = {}
    
    L = len(layers_dims)#网络层数
    
    for l in range(1,L):
       parameters["W" + str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
       parameters["b" + str(l)] = np.zeros((layers_dims[l],1))
       
       #使用断言确保我的数据格式是正确的
       assert(parameters["W" + str(l)].shape == (layers_dims[l],layers_dims[l-1]))
       assert(parameters["b" + str(l)].shape == (layers_dims[l],1))
       
    return parameters

##测试初始化为0
parameters = initialize_parameters_zeros([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

##用上述初始化为0 训练模型

parameters = model(train_X,train_Y,initialization="zeros",is_plot=True)

##用上述初始化0看一下预测
print("训练集：")
predictions_train = init_utils.predict(train_X,train_Y,parameters)
print("测试集：")
predictions_test = init_utils.predict(test_X,test_Y,parameters)

print("train_Y:" +str(train_Y))
print("predictions_train = " + str(predictions_train))
print("test_Y:" +str(test_Y))
print("predictions_test = " + str(predictions_test))

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
init_utils.plot_decision_boundary(lambda x:init_utils.predict_dec(parameters,x.T),train_X,train_Y)

#####随机初始化

def initialize_parameters_random(layers_dims):
    """
    参数：
        layers_dims - 列表，模型的层数和对应每一层的节点的数量
    返回：
        parameters - 包含了所有W和b的字典
            W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
            b1 - 偏置向量，维度为（layers_dims[1],1）
            ···
            WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
            b1 - 偏置向量，维度为（layers_dims[L],1）
    """
    
    np.random.seed(3)   #指定随机种子
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l - 1]) * 10 ##使用10倍的缩放
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))
        
        #使用断言 确保数据格式正确
        assert(parameters['W'+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters['b'+str(l)].shape == (layers_dims[l],1))
    
    return parameters

##测试一下
parameters = initialize_parameters_random([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

###看模型的运行结果
parameters = model(train_X,train_Y,initialization = "random",is_plot=True)
print("训练集：")
predictions_train = init_utils.predict(train_X,train_Y,parameters)
print("测试集：")
predictions_test = init_utils.predict(test_X,test_Y,parameters)

print(predictions_train)
print(predictions_test)


"""
当前的图表和子图可以使用plt.gcf()和plt.gca()获得
分别表示Get Current Figure和Get Current Axes。
"""

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
##使用过大的随机数初始化回减慢优化的速度

####抑制梯度异常初始化
#我们会用到√(2/上一层的维度)这个公式来初始化参数

def initialize_parameters_he(layers_dims):
    """
    参数：
        layers_dims - 列表，模型的层数和对应每一层的节点的数量
    返回
        parameters - 包含了所有W和b的字典
            W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
            b1 - 偏置向量，维度为（layers_dims[1],1）
            ···
            WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
            b1 - 偏置向量，维度为（layers_dims[L],1）
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))
        
        #使用断言确保数据格式正确
        assert(parameters["W"+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape == (layers_dims[l],1))
    
    return parameters

#测试函数
parameters = initialize_parameters_he([2,4,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

##对模型实际运行一下
parameters = model(train_X,train_Y,initialization = "he",is_plot=True)
print("训练集：")
predictions_train = init_utils.predict(train_X,train_Y,parameters)
print("测试集：")
predictions_test = init_utils.predict(test_X,test_Y,parameters)

##绘制预测情况
plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
init_utils.plot_decision_boundary(lambda x : init_utils.predict_dec(parameters,x.T),train_X,train_Y)

"""
总结：
1、不同的初始化方法可能导致性能最终不同
2、随机初始化有助于打破对称，使得不同隐藏层的单元学习到不同的参数
3、初始化时，初始值不宜过大
4、He初始化搭配ReLU激活函数常常可以得到不错的效果
"""

###在数据集没有足够大的情况下，可能会发生过拟合，就是在训练集上的精确度很好，但是在遇到新的样本时，精确度会下降
###接下来降实现正则化，防止出现过拟合




















