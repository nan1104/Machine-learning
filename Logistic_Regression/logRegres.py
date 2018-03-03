#coding:utf-8
#Logistic回归与Sigmoid函数
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    #从testSet.txt导入数据集和标签集
    dataMat = [];labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  #每行按\t分割
        dataMat.append([1.0 , float(lineArr[0]) , float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat , labelMat

def sigmoid(inX):
    #sigmoid函数，输出一个0-1之间的数
    return 1.0/(1+exp(-inX))

#梯度上升算法
def gradAscent(dataMatIn , classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose() #转换为Numpy数据类型
    m , n = shape(dataMatrix) #m为行数，即样本数，n为列数，即特征数
    alpha = 0.001 #学习率
    maxCycles = 500 #最大迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

#随机梯度上升算法
def stocGradAscent0(dataMatrix , classsLabels):
    dataMatrix = array(dataMatrix)  #不加上这行会报错TypeError: 'numpy.float64' object cannot be interpreted as an integer
    m , n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classsLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#改进后的随机梯度上升法
def stocGradAscent1(dataMatrix , classsLabels , numIter = 150):
    dataMatrix = array(dataMatrix)  #不加上这行会报错TypeError: 'numpy.float64' object cannot be interpreted as an integer
    m , n = shape(dataMatrix) #m样本数
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + i + j) + 0.01 #alpha每次迭代时进行调整,缓解数据波动或者高频波动
            randInx = int(random.uniform(0,len(dataIndex))) #随机迭代更新，减少周期性波动
            h = sigmoid(sum(dataMatrix[randInx]*weights))
            error = classsLabels[randInx] - h
            weights = weights + alpha * error * dataMatrix[randInx]
            del(dataIndex[randInx])
    return weights

def plotBestFit(wei):
    #weights = wei.getA() #将矩阵转换为array
    dataMat , labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [] ; ycord1 = []
    xcord2 = [] ; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1]);ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0 , 3.0 , 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2] #最佳拟合曲线,设定0=w0x0 + w1x1 + w1x1 + w2x2 , x0 = 1 , 解出x1和x2的关系
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

if __name__ == "__main__":
    dataArr , labelMat = loadDataSet()
    weights = stocGradAscent1(dataArr , labelMat)
    print(plotBestFit(weights))