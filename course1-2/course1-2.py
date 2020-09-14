# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:50:55 2020

@author: FMENG
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset


train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()

index = 25
plt.imshow(train_set_x_orig[index])

#打印出当前训练值标签 y
#使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1]，【压缩后】 np.squeece(train_set_y[:,index])的值为1   【1】和数字1
#只有压缩后的值才能进行解码操作

print("y=" + str(train_set_y[:,index]) +",it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+" picture")

m_train = train_set_y.shape[1] #训练集里图片的数量
m_test = test_set_y.shape[1] #测试集里的数量
num_px = train_set_x_orig.shape[1] #训练、测试集里面的图片的宽度和高度（均为64x64）

print("训练集的数量： m_train = " + str(m_train))
print("测试集的数量：m_test = " + str(m_test))
print("每张图片的宽/高： num_px = "+str(num_px))
print("每张图片的大小：(" + str(num_px) +"," + str(num_px) + ",3)")
print("训练集_图片的维数：" + str(train_set_x_orig.shape))  #(209,64,64,3)
print("训练集_标签的维数：" + str(train_set_y.shape))  #(1,209)
print("测试集_图片的维数：" + str(test_set_x_orig.shape)) #(50,64,64,3)
print("测试集_标签的维数：" + str(test_set_y.shape))  #(1,50)

#【每一列代表一个平摊的图像】 把维度64x64x3的numpy数组  重新构造为（64x64x3,1）的数组
#乘3是因为 每张照片是由64x64的像素构成，每个像素由（R,G,B）三原色构成

#X_flatten = X.reshape(X.shape[0],-1).T #X.T 是X的转置
#将训练集的维度降低并转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
#将测试集的维度降低并转置
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print("训练集降维最后的维度：" + str(train_set_x_flatten.shape))
print("训练集—标签的维数：" + str(train_set_y.shape))
print("测试集降维之后的维度：" + str(test_set_x_flatten.shape))
print("测试集—标签的维数：" + str(test_set_y.shape))

#像素值R,G,B在0-255数值之间，所以 将我们的数据除以255，让标准化的数据位于【0，1】之间

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

#建立神经网络的主要步骤
##1、定义模型结构（例如输入特征的数量）
##2、初始化模型的参数
##3、循环：
###3.1 计算当前的损失（正向传播） 计算L(a,y)
###3.2 计算当前梯度（反向传播） 
###3.3 更新参数（梯度下降）

###构建sigmoid
def sigmoid(z):
    """
    参数：
     z : 任何大小的标量或numpy数组
     
    返回：
     s: sigmoid(z)
    """
    s = 1/(1+np.exp(-z))
    return s

##测试sigmoid
    
print("======================测试sigmoid=======================")
print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(9.2) = " + str(sigmoid(9.2)))

##initialize_with_zeros初始化参数w和b

def initialize_with_zeros(dim):
    """
    此函数为w创建一个维度为（dim,1）的0向量，并将b初始化为0
    
    参数：
        dim: 我们想要的w的矢量的大小（或者这种情况下的参数数量）
    
    返回
        w: 维度为（dim,1）的初始化向量
        b: 初始化的标量（对应于偏差）
    """
    w = np.zeros(shape = (dim,1))
    b = 0
    #使用assert来确保我要的数据是正确的
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    
    return(w,b)
    
##计算成本函数及其渐变的函数propagate() 即实现正向与反向传播 
    
def propagate(w,b,X,Y):
    """
    实现前向和后向传播的成本函数及其梯度。
    参数：
        w: 权重，大小不等的数组（num_px*num_px*3,1）
        b: 偏差，一个标量
        X：矩阵类型为（num_px*num_px*3,训练数量）
        Y：真正的“标签”矢量（如果是猫则为1，如果不是猫则为0），矩阵维度为（1，训练数据数量）
        
    返回：
        cost: 逻辑回归的负对数似然成本
        dw： 相对于w的损失梯度，因此与w相同的形状
        db： 相对于b的损失梯度，因此与b的形状相同
        
    """
    m = X.shape[1]
    
    #正向传播
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    
    #反向传播 dz = A -Y; dw = xdz.T; db = dz 
    dw = (1 / m) * np.dot(X,(A-Y).T) 
    db = (1 / m) * np.sum(A - Y)
    
    #使用assert确保我的数据维度正确
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    #创建一个字典，把dw和db保存起来
    
    grads = {
            "dw":dw,
            "db":db
            }
    return (grads,cost)

#测试propagate
print("===================测试propagate=====================")
#初始化参数
w,b,X,Y = np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
grads,cost = propagate(w,b,X,Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))

#更新参数  设置迭代次数 更新w和b

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    """
    此函数通过运行梯度下降算法来优化w和b
    
    参数：
        w:权重，大小不等的数组（num_px*num_px*3,1）
        b:偏差，一个标量
        X:维度为（num_px*num_px*3,训练数据的数量）的数组
        Y:真正的“标签”矢量 （如果是猫则为1，如果不是猫则为0），矩阵维度为（1，训练数据的数量）
        num_iterations:  优化循环的迭代次数
        learning_rate:梯度下降更新规则的学习率
        print_cost:每100步打印一次损失值
        
    返回：
        params: 包含权重w和偏差b的字典
        grads: 包含权重和偏差相对于成本函数的梯度字典
        成本：优化期间计算的所有成本列表，将用于绘制学习曲线
        
        
    提示：
    我们需要写下两个步骤并遍历它们：
        1、计算当前参数的成本和梯度，使用propagate()
        2、使用w和b的梯度下降法则更新参数
    """
    costs = []
    
    for i in range(num_iterations):
        
        grads,cost = propagate(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #记录成本
        if i % 100 == 0:
            costs.append(cost)
        #打印成本数据
        if(print_cost) and (i % 100 == 0):
            print("迭代的次数： %i,误差值：%f" %(i,cost))
        
    
    params = {
            "w" : w,
            "b" : b
            }
    grads = {
            "dw" : dw,
            "db" : db
            }
    return(params,grads,costs)

print("=================测试optimize================")
w,b,X,Y = np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]]) #w为两行一列，X为两行两列，Y为一行两列
params,grads,costs = optimize(w,b,X,Y,num_iterations = 100,learning_rate = 0.009,print_cost = False)
print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = "+ str(grads["db"]))

##optimize函数输出的是已学习的w和b的值 使用w和b的值预测数据集X的标签

##实现预测函数predict() ，计算预测有两个步骤：
###1、计算 Y^ = A = sigmoid(w.T*X + b)
###2、将a的值变为0（如果激活值 <=0.5） 或者为1（如果激活值>0.5）

def predict(w,b,X):
    """
    使用学习逻辑回归参数logistic（w,b）预测标签是0还是1
    
    参数：
        w: 权重，大小不等的数组（num_px*num_px*3,1）
        b: 偏差，一个标量
        X：维度为（num_pX*num_px*3,训练数据的数量）的数据
        
    返回：
        Y_prediction: 包含X中所有图片的所有预测【0|1】的一个numpy数组（向量）
        
    """
    
    m = X.shape[1] #图片的数量
    Y_prediction = np.zeros((1,m))#一行m列
    w = w.reshape(X.shape[0],1)
    
    #计算预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        #将概率a[0,i] 转换为实际预测p[0.i]
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0
    #使用断言
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction

#测试predict
print("===============测试predict==================")
w,b,X,Y = np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
print("predictions: " + str(predict(w,b,X)))

###将所有的函数整合到一个model()函数中，届时只需要调用一个model()
def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate = 0.5,print_cost = False):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    
    参数：
        X_train: numpy的数组，维度为（num_pX*num_px*3,m_train）的训练集
        Y_train：numpy的数组，维度为（1，m_train）（矢量）的训练标签集
        X_test: numpy的数组，维度为（num_pX*num_px*3,m_test）的测试集
        Y_test: numpy的数组，维度为（1，m_test）的（向量）的测试标签集
        num_iterations: 表示用于优化参数的迭代次数的超参数
        learning_rate: 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost: 设置为true以每100次迭代打印成本
    
    返回：
        d:包含有关模型信息的字典
    
    """
    w,b = initialize_with_zeros(X_train.shape[0])
    
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    #从字典“参数”中检索参数w和b
    w,b = parameters["w"],parameters["b"]
    
    #预测测试/训练集的例子
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    
    #打印训练后的准确性
    print("训练集准确性:" , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100),"%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100),"%")
    
    d = {
            "costs":costs,
            "Y_prediction_test":Y_prediction_test,
            "Y_prediction_train":Y_prediction_train,
            "w":w,
            "b":b,
            "learning_rate":learning_rate,
            "num_iterations":num_iterations
            }
    return d

print("=====================测试model=======================")
###这里加载的是真实数据
#d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations = 2000,learning_rate = 0.005,print_cost = True)

###绘制图
"""
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

"""
###进一步分析学习率alpha的可能选择
"""
学习率α决定了更新参数的速度；
如果学习率过高，我们可能超过最优值
如果学习率过低，需要迭代多次才能达到最优值
"""

learning_rates = [0.01,0.001,0.0001]
models = {}
for i in learning_rates:
    print("learning rate is:" + str(i))
    models[str(i)] = model(train_set_x,train_set_y,test_set_x,test_set_y,learning_rate = i,print_cost = False)
    print('\n' + "------------------------------------")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label = str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()









