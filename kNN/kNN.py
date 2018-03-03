#coding:utf-8
#约会网站函数
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
#创建数据
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
#k近邻算法
def classify0(inX , dataSet , labels ,k):
    dataSetSize = dataSet.shape[0] #shape = (4,2) , size = 4
    diffMat = tile(inX , (dataSetSize  , 1)) - dataSet #将inX整形成为dataSet的大小
    sqDiffMat = diffMat **2 #矩阵平方
    sqDistances = sqDiffMat.sum(axis = 1) #水平相加
    distances = sqDistances **0.5 #取根号得距离
    sortedDistIndicies = distances.argsort() #排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #得到距离第i近的点的label
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #D.get(k[,d]) -> D[k] if k in D, else d. d defaults to None.
        #itemgetter(item, ...) --> itemgetter object
        #Return a callable object that fetches the given item(s) from its operand.
        #After f = itemgetter(2), the call f(r) returns r[2].
    sortedClassCount = sorted(classCount.items() , key = operator.itemgetter(1) , reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOLines = len(arrayOLines) #得到文件行数
    returnMat = zeros((numberOLines,3)) #创建返回的Numpy矩阵,保存特征
    classLabelVector = [] #保存最后一列元素
    index = 0
    for line in arrayOLines:
        #解析文件数据到列表
        line = line.strip() #截取掉所有回车字符
        listFromLine = line.split('\t') #用\t将整行数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet / tile(ranges ,(m,1))
    return normDataSet , ranges , minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat , datingLabels = file2matrix('datingTestSet2.txt')
    normMat , ranges , minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) #测试样本数
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with : %d , the real answer is : %d"\
              % (classifierResult , datingLabels[i]))
        if (classifierResult != datingLabels[i]) : errorCount += 1.0
    print("the total error rate is : %f" % (errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMile = float(input("frequent fliter miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat , datingLabels = file2matrix('datingTestSet2.txt')
    normMat , ranges , minVals = autoNorm(datingDataMat)
    inArr = array([ffMile , percentTats , iceCream])
    classifierResult = classify0((inArr - minVals)/ranges , normMat , datingLabels ,3)
    print("You will probably like this this person :" , resultList[classifierResult - 1])

classifyPerson()