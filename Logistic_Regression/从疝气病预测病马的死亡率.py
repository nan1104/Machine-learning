#coding:utf-8
from numpy import *

def sigmoid(inX):
    #sigmoid函数，输出一个0-1之间的数
    return 1.0/(1+exp(-inX))

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

def  classifyVector(inX , weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5 : return 1.0
    else : return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [] ; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),array(trainingLabels),500)
    errorCount = 0;numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if (int)(classifyVector(array(lineArr),trainWeights))!=  int(currLine[21]) :
            errorCount += 1
    errorRate = float(errorCount)/numTestVec
    print("the error rate of this test is : %f " % errorRate)
    return errorRate

def multiTest():
    numTests = 10 ; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is : %f" % (numTests , errorSum/float(numTests)))

if __name__ == "__main__":
    multiTest()