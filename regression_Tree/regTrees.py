#coding:utf-8
"""
利用CART(Classification And Regression Trees,分类回归树)的树构建算法
CART假设决策树是二叉树，内部结点特征的取值为“是”、“否”
决策树的生成就是递归地构建二叉决策树的过程
对回归树用平方误差最小化准则，对分类树用基尼指数（Gini index）最小化准则
这里实现的是平方误差最小化回归树 和 模型树
"""
from numpy import *
import matplotlib.pyplot as plt

class  treeNode():
    def __init__(self , feat , val , right , left):
        featureToSplition =  feat #特征
        valueOfSplit = val #特征值
        rightBranch = right
        leftBranch = left

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split()
        # python3不适用：fltLine = map(float,curLine) 修改为：
        fltLine = list(map(float, curLine))  # 将每行映射成浮点数，python3返回值改变，所以需要
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet , feature , value):
    """
    :param dataSet:输入数据集合
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return: 切分后的两个数据集合

    nonzero(X)返回根据特定条件X筛选出来的数据点，返回的形式为元组
    Returns a tuple of arrays, one for each dimension of `a`
    所以nonzero(dataSet[:,feature] > value)[0],[0]代表取元组的第一个值，即矩阵的行数
    """
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0] ,:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0] ,:]
    return mat0 , mat1

def regLeaf(dataSet):
    """
    :param dataSet:
    :return: 数据集的均值
    当chooseBestSplit()函数确定不再对数据进行切分时，调用该函数来得到叶节点的模型，在回归树种，该模型就是目标变量的均值
    """
    return mean(dataSet[:,-1])

def regErr(dataSet):
    """
    :param dataSet:
    :return: 目标变量的平方误差
    var()为均方差函数，乘以样本数即为总方差
    """
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):
    m , n = shape(dataSet)
    X = mat(ones((m,n))) ; Y = mat(ones((m,1))) #将X与Y中的数据格式化
    X[:,1:n] = dataSet[:,0:n-1] ; Y = dataSet[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular , cannot do inverse,\ntry increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws , X ,Y

def modelLeaf(dataSet):
    ws , X , Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws , X , Y =linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat , 2))

def chooseBestSplit(dataSet , leafType = regLeaf , errType = regErr , ops=(1,4)):
    """
    找到数据的最佳二元切分方式
    :param dataSet: 数据矩阵
    :param leafType: 生成叶节点的方式——均值
    :param errType: 计算误差的方式——总方差
    :param ops: ops的两个参数由用户指定来控制函数的停止时机,第一个参数为tolS:如果误差减少小于tolS则退出，
    第二个参数tolN:如果划分后的数据集数目小于tolN则退出
    :return:
    """
    tolS = ops[0] ; tolN = ops[1]
    if (len(set(dataSet[:,-1].T.tolist()[0])) == 1): #如果所有值相等则退出
        return None , leafType(dataSet)
    m , n = shape(dataSet) #m是样本数，n是特征数
    S = errType(dataSet) #总方差
    bestS = inf ; bestIndex = 0 ; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]): #遍历该特征的每一个取值
            mat0 , mat1 = binSplitDataSet(dataSet , featIndex , splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN) : continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S - bestS) < tolS : #如果误差减少不大则退出
        return None,leafType(dataSet)
    mat0 , mat1 = binSplitDataSet(dataSet , bestIndex , bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #如果切分出的数据集很小就退出
        return None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet , leafType = regLeaf , errType = regErr , ops=(1,4)):
    feat , val = chooseBestSplit(dataSet , leafType , errType , ops)
    if feat == None: return val #满足停止条件时返回叶节点值
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet , rSet = binSplitDataSet(dataSet , feat , val)
    retTree['left'] = createTree(lSet , leafType , errType , ops)
    retTree['right'] = createTree(rSet , leafType , errType , ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']) : tree['right'] = getMean(tree['right'])
    if isTree(tree['left']) : tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left'])/2.0

def prune(tree , testData):
    if shape(testData)[0] == 0 : return getMean(tree) #没有测试数据则对树进行塌陷处理
    if (isTree(tree['right']) or isTree(tree['left'])): #如果左右分支有一个不是叶节点，则进行划分
        lSet , rSet = binSplitDataSet(testData , tree['spInd'] , tree['spVal'])
    if isTree(tree['left']) : tree['left'] = prune(tree['left'] , lSet) #如果不是叶节点，则递归调用prune
    if isTree(tree['right']) : tree['right'] = prune(tree['right'] , rSet)
    if not isTree(tree['left']) and not isTree(tree['right']): #当左右节点都是叶子时，进行合并判断
        lSet , rSet = binSplitDataSet(testData , tree['spInd'] , tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'] , 2)) + sum(power(rSet[:,-1] - tree['right'] , 2)) #合并前的误差
        treeMean = (tree['left'] + tree['right'])/2.0 #合并后的均值
        errorMerge = sum(power(testData[:,-1] - treeMean , 2)) #合并后的误差
        if errorMerge < errorNoMerge : #如果合并后误差减小，则将两个叶节点进行合并
            print("merging")
            return treeMean #返回合并后的叶节点的值
        else : return tree
    else : return tree

#1-回归树
def regTreeEval(model , inDat):
    return float(model)

#2-模型树
def modelTreeEval(model , inDat):
    n = shape(inDat)[1] #这个数据集中n=1
    X = mat(ones((1,n+1))) #X的shape=(1,2)
    X[:,1:n+1] = inDat #X=[1,x]
    return float(X*model) #返回ws*x

def treeForeCast(tree , inData , modelEval=regTreeEval):
    """
    自顶向下遍历整棵树，直至命中叶节点为止
    :param tree: 树
    :param inData: 输入数据,[[x]]
    :param modelEval:
    :return:
    """
    if not isTree(tree): return modelEval(tree , inData) #如果是叶节点，计算预测值
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'] , inData , modelEval)
        else:
            return modelEval(tree['left'] , inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'] , inData , modelEval)
        else:
            return modelEval(tree['right'] , inData)

def createForeCast(tree , testData , modelEval=regTreeEval):
    #输入tree和X，计算函数预测值
    m = len(testData) #测试样本数
    yHat = mat(zeros((m,1))) #函数预测值
    for i in range(m):
        #计算每一个函数预测值
        yHat[i,0] = treeForeCast(tree , mat(testData[i]) , modelEval)
    return yHat

if __name__ == "__main__":
    """
    myDat = loadDataSet('ex2.txt')
    myMat = mat(myDat)
    myTree = createTree(myMat , ops=(1,4))
    myDatTest = loadDataSet('ex2test.txt')
    myMatTest = mat(myDatTest)
    plt.plot(myMat[:, 0], myMat[:, 1],'ro')
    plt.show()
    print(myTree)
    print(prune(myTree , myMatTest))
    myMat2 = mat(loadDataSet('exp2.txt'))
    myTree2 = createTree(myMat2 , modelLeaf , modelErr)
    print(myTree2)
    """
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat , ops=(1,20))
    print(myTree)
    yHat = createForeCast(myTree , testMat[:,0])
    print(corrcoef(yHat , testMat[:,1] , rowvar=0)[0,1])
    myTree = createTree(trainMat , modelLeaf , modelErr, (1,20))
    yHat = createForeCast(myTree , testMat[:,0], modelTreeEval)
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
    print(shape(testMat[:,0]))
    a = testMat[:,0]
    print(mat(a[1]))
    b = a[1]
    print(shape(b))