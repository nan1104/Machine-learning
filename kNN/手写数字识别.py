#coding:utf-8
from numpy import *
from os import listdir
def img2vector(filename):
    #将32*32的图片转换为1*1024的向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readlines()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

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

def handwirtingClassTest():
    hwLabels = []
    #获取目录内容
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #从文件名解析分类数字
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(m):
        fileNameStr = testFileList[i]
        #从文件名解析分类数字
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with %d ,the real answer is %d' % (classifierResult , classNumStr))
        if(classifierResult != classNumStr) : errorCount += 1.0
    print("\nthe total number of errors is %d" % errorCount)
    print("\nthe total error rate is %f " % (errorCount/float(mTest)))

handwirtingClassTest()