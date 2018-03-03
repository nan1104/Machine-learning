#coding:utf-8
#ID3算法用于构建离散数据的决策树
from math import log
import decision_Tree.treePlotter
import operator
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #计算数据集中实例的总数
    labelCounts = {}
    #为所有可能的类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    #以2为底求对数
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfing','flippers']
    return dataSet , labels

def splitDataSet(dataSet , axis , value):
    retDataSet = []  #创建新的list对象
    for featVec in dataSet:
        if featVec[axis] == value:
            #抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #创建唯一的分类标签列表
        featList = [exam[i] for exam in dataSet] #通过列表推导，将第i个特征所有可能存在的值或所有可能的值写入新list中
        uniqueVals = set(featList) #通过原声的集合set数据类型得到列表中唯一元素值的最快方法，得到特征i的所有取值
        newEntropy = 0.0
        for value in uniqueVals:
            #计算每种划分方式下的信息熵
            subDataSet = splitDataSet(dataSet , i , value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys() : classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),\
                              key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet , labels):
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止继续划分，直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #使用完了所有特征，扔不能划分为唯一类别的分组
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel :{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(\
            dataSet,bestFeat,value),subLabels)
    return myTree

def classify(inputTree , featLabels ,testVec):
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) #将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if(type(secondDict[key]).__name__=='dict'):
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree , filename):
    import pickle
    fw = open(filename , 'wb')
    pickle.dump(inputTree , fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

fr = open('lenses.txt')
lenses = [inst.strp().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLabels)
print(lensesTree)
decision_Tree.treePlotter.createPlot(lensesTree)