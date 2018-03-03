#coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

#数据导入函数
def loadDataSet (fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = [] ; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat , labelMat

#标准回归函数,用来计算最佳拟合直线
def standRegres(xArr , yArr):
    xMat = mat(xArr) ; yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0: #numpy提供一个线性代数的库linalg，可以直接调用linalg.det()计算行列式
        print("This matrix is singular , cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def plotData():
    filename = 'ex0.txt'
    xArr, yArr = loadDataSet(filename)
    xMat = mat(xArr);
    yMat = mat(yArr)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    # 取第二个特征绘图
    # flatten()函数转化成一维矩阵
    # matrix.A[0]属性返回矩阵变成的数组，和getA()方法一样
    '''下面这段代码绘制原始数据的散点图'''
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    '''下面这段代码绘制拟合直线'''
    # 返回给定数据的数组形式的拷贝hvoooooooooooooooooooooo
    xCopy = xMat.copy()
    xCopy.sort(0)
    weights = standRegres(xArr, yArr)
    yHat = xCopy * weights  # yHat 表示拟合直线的纵坐标，用回归系数求出
    ax.plot(xCopy[:, 1], yHat, c='green')
    plt.show()
    """Numpy提供了相关系数的计算方法：可以通过命令corrcoef(yEstimate,yActual)来计算预测值和真实值的相关性"""
    yHat = xMat * weights
    corrcoef(yHat.T , yMat) #这时需要保证两个向量都是行向量
    print(corrcoef(yHat.T , yMat))

if __name__ == "__main__":
    plotData()