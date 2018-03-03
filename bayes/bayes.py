#coding:utf-8
#朴素贝叶斯应用：过滤网站的恶意留言
#该模型是一个词集模型(set-of-words model)，仅将每个词的出现与否作为一个特征
#而词袋模型中每个单词可以出现多次
from numpy import *

#词表到向量的转换函数
def loadDataSet():
    #函数创建一些实验样本，返回第一个变量是进行此条切分后的文档集合，第二个变量是类别标签的集合
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0 , 1 , 0 , 1 , 0 , 1] #1 代表侮辱性文字，0 代表正常言论
    return postingList , classVec

def createVocabList(dataSet):
    #创建一个包含在所有文档中出现的不重复词的列表
    vocabSet = set([]) #创建一个空集
    for document in dataSet :
        vocabSet = vocabSet | set(document) #创建两个集合的并集
    return list(vocabSet)

#词集模型中的函数
def setOfWords2Vec(vocabList , inputSet):
    #函数的输入参数为词汇表及某个文档，输出文档向量，向量每一个元素为0或者1，表示词汇表中的单词在输入文档中是否出现
    #创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:print("the word : %s is not in my Vocabulary !" % word)
    return returnVec

#词袋模型中的函数
def bagOfWords2Vec(vocabList , inputSet):
    #函数的输入参数为词汇表及某个文档，输出文档向量，向量每一个元素为0或者1，表示词汇表中的单词在输入文档中是否出现
    #创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:print("the word : %s is not in my Vocabulary !" % word)
    return returnVec

#朴素贝叶斯分类器训练函数
def trainNBO(trainMatrix , trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords);p1Num = ones(numWords)
    p0Denom = 2.0;p1Denom = 2.0
    for i in range(numTrainDocs):
        #x向量相加
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #对每个元素做除法
    p1Vect = log(p1Num/p1Denom) #change to log()
    p0Vect = log(p0Num/p0Denom) #change to log()
    return p0Vect , p1Vect , pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify , p0Vec , p1Vec , pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1) #相当于是后验概率：概率密度乘以先验概率
    p0 = sum(vec2Classify*p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    #便利函数，封装所有操作，以节省输入代码的时间
    listOPosts , listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V , p1V , pAb = trainNBO(array(trainMat) , array(listClasses))
    testEntry = ['love' , 'my' , 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList , testEntry))
    print(testEntry , 'classified as :' ,classifyNB(thisDoc , p0V ,p1V ,pAb))
    testEntry = ['stupid' , 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()