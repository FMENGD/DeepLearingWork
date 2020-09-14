# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:48:25 2020

@author: FMENG
"""

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets


np.random.seed(1) #设置一个固定的随机种子

X,Y = load_planar_dataset()

#plt.scatter(X[0,:], X[1,:], c=Y, s=40, cmap=plt.cm.Spectral) #绘制散点图 有错误

plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)#绘制散点图

"""
数据是一朵红色（y=0）和一些蓝色（y=1）的数据点
目标：建立一个模型来适应这些数据
X：一个numpy的矩阵，包含了这些数据点的数值
Y：一个numpy的向量，对应着的是X的标签【0|1】
"""

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1] #训练集里面的数量

print("X的维度为：" + str(shape_X))
print("Y的维度为：" + str(shape_Y))
print("数据集里面的数据有：" + str(m) + "个")

#简单逻辑回归的分类效果

clf = sklearn.linear_model.LogisticRegressionCV() #逻辑回归模型
clf.fit(X.T,Y.T) #用训练数据拟合模型

#绘制逻辑回归分类器

plot_decision_boundary(lambda x: clf.predict(x), X, np.squeeze(Y)) #绘制决策边界
plt.title("Logistic Regression") #图标题
LR_predictions  = clf.predict(X.T) #预测结果
print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) + 
        np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       "% " + "(正确标记的数据点所占的百分比)")

"""
构建神经网络的一般方法：
1、定义神经网络结构
2、初始化模型的参数
3、循环：
1）实施前向传播
2）计算损失
3）实现后向传播
4）更新参数

将它们合并到一个model()函数中，构建好莫得了() 并学习正确的参数，
就可以预测新的数据了
"""

##定义神经网络结构
def layer_sizes(X,Y):
    """
    参数：
    X - 输入数据集，维度为（输入的数量，训练/测试的数量）
    Y - 标签，维度为（输出的数量，训练/测试数量）
    
    返回：
    n_x - 输入层的数量
    n_h - 隐藏层的数量
    n_y - 输出层的数量
    """
    
    n_x = X.shape[0]#输入层的节点个数
    n_h = 4#隐藏层的节点格式
    n_y = Y.shape[0]#输出层的节点个数
    
    return(n_x,n_h,n_y)

print("===============测试layer_sizes=======")
X_asses,Y_asses = layer_sizes_test_case()
(n_x,n_h,n_y) = layer_sizes(X_asses,Y_asses)
print("输入层的节点数量为： n_x = " + str(n_x))
print("隐藏层的节点数量为： n_h = " + str(n_h))
print("输出层的节点数量为： n_y = " + str(n_y))

##初始化模型的参数
def initialize_parameters(n_x,n_h,n_y):
    """
    参数：
        n_x - 输入层节点的数量
        n_h - 隐藏层节点的数量
        n_y- 输出层节点的数量
    
    返回：
        parameters -  包含参数的字典：
            W1 - 权重矩阵，维度为（n_h,n_x）
            b1 - 偏向量，维度为（n_h,1）
            W2 - 权重矩阵，维度为（n_y,n_h）
            b2 - 偏向量，维度为（n_y,1）
                
    """
    np.random.seed(2)#指定一个随机种子，以便你的输出和我们的一样
    W1 = np.random.randn(n_h,n_x) *  0.01
    b1 = np.zeros(shape=(n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros(shape=(n_y,1))
    
    #使用断言 确保数据格式正确
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    
    parameters = {
            "W1" : W1,
            "b1" : b1,
            "W2" : W2,
            "b2" : b2
            }
    return parameters

#测试initialize——parameter
print("============测试initialize_parameters========")
n_x,n_h,n_y = initialize_parameters_test_case() #2,4,1
parameters = initialize_parameters(n_x,n_h,n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

"""
循环！
前向传播函数 forwaard_propagation()
可以使用sigmoid（）函数 也可以使用np.tanh()函数
步骤：
 使用字典类型的parameters（它是initialize_parameters()的输出） 检索每个参数
 实现向前传播，计算Z[1],A[1],Z[2],A[2]
 反向传播所需的值存储在cache中，cache尖作为反向传播函数的输入
 
"""
def forward_propagation(X,parameters):
    """
    参数： 
        X - 维度为（n_x,m）的输入数据
        parameters - 初始化函数（initialize_parameters）的输入
    返回：
        A2 - 使用sigmoid（）函数计算的第二次激活函数后的数值
        cache - 包含“Z1”,"A2"，“Z2”和“A2”的字典类型变量
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    #前向传播计算A2
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    #使用断言确保数据格式正确
    assert(A2.shape == (1,X.shape[1]))
    cache = {
            "Z1" : Z1,
            "A1" : A1,
            "Z2" : Z2,
            "A2" : A2
            }
    
    return(A2,cache)
##测试forward——propagation
print("=====================测试forward_propagation==========================")
X_assess,parameters = forward_propagation_test_case()
A2,cache = forward_propagation(X_assess,parameters)
print(np.mean(cache["Z1"]),np.mean(cache["A1"]),np.mean(cache["Z2"]),np.mean(cache["A2"]))

##计算成本
"""
成本公式 J = -1/m ∑（y(i)log(a[2](i)) + (1 - y)log(1-a[2](i))）

"""
def compute_cost(A2,Y,parameters):
    """
    计算上述成本方程中给出的交叉熵成本
    
    参数：
        A2 - 使用sigmoid（）函数计算的第二次激活后的数值
        Y - "True"标签向量，维度为（1，数量）
        parameters - 一个包含W1，B1，W2和B2的字典类型的变量
    返回：
        成本 - 交叉熵成本给出的方程
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    #计算成本
    logprobs = logprobs = np.multiply(np.log(A2),Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    
    assert(isinstance(cost,float))
    
    return cost

#测试成本函数
print("====================测试compute_cost-====================")
A2,Y_assess,parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2,Y_assess,parameters)))

###后向传播
#六大公式  dz[2] dw[2] db[2] dz[1] dw[1] db[1]
def backward_propagation(parameters,cache,X,Y):
    """
    使用上述说明搭建反向传播网络
    
    参数：
        parameters - 包含我们的参数的一个字典类型的变量
        cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量
        X - 输入数据，维度为（2，数量）
        Y - “True”标签，维度为（1，数量）
        
    返回：
        grads - 包含W和b的导数 一个字典类型的变量
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    db2 = (1/m) * np.sum(dZ2,axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),1 - np.power(A1,2))
    dW1 = (1/m) * np.dot(dZ1,X.T)
    db1 = (1/m) * np.sum(dZ1,axis=1,keepdims=True)
    
    grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
            }
    return grads

##测试反向函数
print("================测试backward_propagation===============")
parameters,cache,X_assess,Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters,cache,X_assess,Y_assess)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))
    
##更新参数

#我们需要使用（dw1,db1,dw2,db2）来更新（W1，b1,W2,b2）

def update_parameters(parameters,grads,learning_rate = 1.2):
    """
    使用上面给出的梯度下降更新规则更新参数
    
    参数：
        parameters - 包含参数的字典类型的变量
        grads - 包含导数值的字典数据类型的变量
        learning_rate - 学习速率
    返回：
        parameters - 包含更新参数的字典数据类型变量
    """
    W1,W2 = parameters["W1"],parameters["W2"]
    b1,b2 = parameters["b1"],parameters["b2"]
    
    dW1,dW2 = grads["dW1"],grads["dW2"]
    db1,db2 = grads["db1"],grads["db2"]
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {
            "W1" : W1,
            "b1" : b1,
            "W2" : W2,
            "b2" : b2
            }
    return parameters

#测试update_parameters
   
print("===============测试update_parameters=================")
parameters,grads = update_parameters_test_case()
parameters = update_parameters(parameters,grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

###整合  将所有东西整合到nn_model()中

def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    """
    参数：
        X - 数据集，维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层节点的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值
    返回：
     parameters - 模型学习的参数，它们可以用来预测
     
     """
     
    np.random.seed(3)#指定随机种子
    n_x = layer_sizes(X,Y)[0] #上面定义的函数  神经网络的结构
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y) #前面定义的初始化神经网络的参数  返回 W1，b1,W2,b2
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters) #######得到cost函数 成本
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate = 0.5)#得到W1，b1,W2,b2
        
        if print_cost:
            if i%1000 == 0:
                print("第",i,"次循环，成本为：" + str(cost))
    return parameters

#测试model
print("==========================测试nn_model==============")
X_asses,Y_asses = nn_model_test_case()

parameters = nn_model(X_assess,Y_assess,4,num_iterations=10000,print_cost=False)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

##预测
#后见predict()来使用模型进行预测，使用前向传播来预测结果

def predict(parameters,X):
    """
    使用学习的参数，为X中的每个示例预测一个类
    
    参数：
        parameters - 包含参数的字典数据类型的变量
        X - 输入数据（n_x,m）
        
    返回：
        predictions - 我们模型预测的向量（红色：0|蓝色：1）
    """
    A2,cache = forward_propagation(X,parameters) #前向返回的是A2和cache
    predictions = np.round(A2) ###np.round 表示四舍五入的值
    
    return predictions

##测试predict
print("===============测试predict=============")

parameters,X_assess = predict_test_case()

predictions = predict(parameters,X_assess)
print("预测的平均值 = " + str(np.mean(predictions)))

###正式运行

parameters = nn_model(X,Y,n_h = 4, num_iterations=10000,print_cost=True)

#绘制边界

plot_decision_boundary(lambda x: predict(parameters,x.T),X,np.squeeze(Y))
plt.title("Decision Boundary for hidden layer size" + str(4))

predictions = predict(parameters,X)
print('准确率：%d' % float((np.dot(Y,predictions.T) + np.dot(1 -Y,1 - predictions.T)) / float(Y.size) * 100) + '%')

###更改隐藏层节点的数量

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] #隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))

    
        

    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






