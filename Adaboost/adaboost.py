#coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

def loadSimpData():
    datMat = matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classLabels = [1.0 , 1.0 , -1.0 , -1.0 , 1.0]
    return datMat , classLabels

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) #列数目
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i])) #一行的特征
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1])) #一行的标签
    return dataMat , labelMat

def stumpclassify(dataMatrix , dimen , threshVal , threshIneq):
    #用于测试是否有某个值小于或者大于我们正在测试的阈值
    retArray = ones((shape(dataMatrix)[0] , 1)) #形状为n*1
    #看第dimen维的数据情况
    if threshIneq == 'It':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr , classLabels ,D):
    """
    在一个加权数据集中循环，并找到具有最低错误率的单层决策树,即弱学习器
    :param dataArr: 数据数组
    :param classLabels: 标签数组
    :param D: 基于数据的权值向量
    :return: 最佳分类器对应的单层决策树字典
    """
    dataMatrix = mat(dataArr) ; labelMat = mat(classLabels).T #将两个数据转为矩阵
    m , n = shape(dataMatrix)#m是数据数量，n是数据维数
    numSteps = 10.0 #用于在特征的所有可能只上进行遍历
    bestStump = {} #该字典用于存储给定权值向量D时所得到的最佳单层决策树的相关信息
    bestClassSet = mat(zeros((m,1)))
    minError = inf
    for i in range(n): #对数据集中的每一个特征（第一层循环）
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps #根据最大值和最小值确定步长
        for j in range(-1 , int(numSteps)+1): #对每个步长
            for inequal in ['It','gt']: #对每个不等号
                threshVal = (rangeMin +float(j)*stepSize)  #当前情况下的阈值
                predictedVals = stumpclassify(dataMatrix , i , threshVal , inequal) #预测变量
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0 #预测对的为0，预测错的为1
                weightedError = D.T*errArr
                #print("split:dim%d,thresh %.2f,thresh inequal:%s,\
                    #the weighted error is %.3f" % \
                      #(i, threshVal , inequal , weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassSet = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump , minError , bestClassSet

def adaBoostTrainDS(dataArr , classLabels , numIt=40):
    #基于单层决策树的AdaBoost训练过程
    weakClassArr = [] #弱分类器向量
    m = shape(dataArr)[0] #样本数
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1))) #记录每个数据点的类别估计累计值
    for i in range(numIt): #循环结束条件：运行numIt次或者知道训练错误率为0为止
        bestStump , error , classEst = buildStump(dataArr , classLabels , D) #建立当前D分布下的单层决策树
        print("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16))) #根据error计算alpha值
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T , classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        #为下一次迭代计算D
        aggClassEst += alpha*classEst
        print("aggClassEst:",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T , ones((m,1)))
        errorRate = aggErrors.sum()/m
        #错误率累加计算
        print("total error:",errorRate,"\n")
        if errorRate == 0.0 : break
    return weakClassArr , aggClassEst

def adaClassify(datToClass , classifierArr):
    """
    AdaBoost分类函数
    :param datToClass:一个或者多个待分类样例
    :param classifierArr: 多个弱分类器组成的数组
    :return: 分类函数
    """
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0] #确定待分类样例的个数m
    aggClassEst = mat(zeros((m,1))) #0列向量
    for i in range(len(classifierArr)):
        classEst = stumpclassify(dataMatrix , classifierArr[i]['dim'],classifierArr[i]['thresh'] , classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst #对每个单层决策树根据alpha的值加权
        print(aggClassEst)
    return sign(aggClassEst) #返回aggClassEnt的符号，即如果大于0返回1，小于0返回-1

def plotROC(predStrengths , classLabels):
    """
    ROC曲线的绘制及QUC计算函数
    ROC曲线，横轴为伪正例的比例，纵轴为真正例的比例
    :param predStrengths: 一个Numpy数组或者一个行向量组成的矩阵，代表分类器的预测强度，在分类器和训练函数将这些数值应用到sign()函数之前，他们就产生了
    :param classLabels:实际的标签
    :return:
    """
    cur = (1.0 , 1.0) #浮点数二元组，保留绘制光标的位置
    ySum = 0.0 #用于计算AUC的值
    numPosClass = sum(array(classLabels) == 1.0) #正例的数目
    yStep = 1/float(numPosClass)  #y坐标轴上的步进数目，1/正例数目
    xStep = 1/float(len(classLabels) - numPosClass) #x坐标轴上的步进数目,1/负例数目
    sortedIndies = predStrengths.argsort() #获得排好序的索引，但这些索引是按从最小到最大排序的，因此需要从点（1,1）开始绘
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111) #用于构建画笔
    for index in sortedIndies.tolist()[0]: #在所有排序值上进行循环
        if classLabels[index] == 1.0: #每得到一个标签为1.0的类则要沿着y轴的方向下降一个步长，即不断降低真阳率
            delX = 0 ; delY = yStep
        else: #类似地，对于每个其他的标签，则是在x方向上倒退了一个步长（假阴率方向）
            delX = xStep ; delY = 0;
            ySum += cur[1]
        ax.plot([cur[0] , cur[0]-delX] , [cur[1] , cur[1]-delY] , c='b')
        cur = (cur[0] - delX , cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Adaboost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("The Area Under the Curve is:",ySum*xStep)

if __name__ == "__main__":
    #datMat , classLabels = loadSimpData()
    #D = mat(ones((5,1))/5)
    #buildStump(datMat,classLabels,D)
    #classifierArray = adaBoostTrainDS(datMat , classLabels ,9)
    #print(adaClassify([[5,5],[0,0]],classifierArray))
    #datArr ,labelArr = loadDataSet('horseColicTraining2.txt')
    datArr = [[0,1,3] , [0,3,1],[1,2,2],[1,1,3],[1,2,3],[0,1,2],[1,1,2],[1,1,1],[1,3,1],[0,2,1]]
    labelArr = [-1,-1,-1,-1,-1,-1,1,1,-1,-1]
    classifierArray , aggClassEst = adaBoostTrainDS(datArr , labelArr)
    #testArr , testLabelArr = loadDataSet('horseColicTest2.txt')
    #prediction10 = adaClassify(testArr , classifierArray)
    #To get the number of misclassified examples type in:
    #errArr = mat(ones((67,1)))
    #print(errArr[prediction10 != mat(testLabelArr).T].sum()) #总共错分样本数目
    plotROC(aggClassEst.T , labelArr)
