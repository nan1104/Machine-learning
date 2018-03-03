#coding:utf-8
#应用：利用朴素贝叶斯过滤垃圾邮件
from numpy import *
import re

def createVocabList(dataSet):
    #创建一个包含在所有文档中出现的不重复词的列表
    vocabSet = set([]) #创建一个空集
    for document in dataSet :
        vocabSet = vocabSet | set(document) #创建两个集合的并集
    return list(vocabSet)

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

def setOfWords2Vec(vocabList , inputSet):
    #函数的输入参数为词汇表及某个文档，输出文档向量，向量每一个元素为0或者1，表示词汇表中的单词在输入文档中是否出现
    #创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:print("the word : %s is not in my Vocabulary !" % word)
    return returnVec

#朴素贝叶斯分类函数
def classifyNB(vec2Classify , p0Vec , p1Vec , pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1) #相当于是后验概率：概率密度乘以先验概率
    p0 = sum(vec2Classify*p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#input is big string, #output is word list
def textParse(bigString):
    listOfTokens = re.split(r'\W+' , bigString)
    return [tok.lower() for tok in listOfTokens if (len(tok) > 1)]

#完整的垃圾邮件测试函数
def spamTest():
    docList = [];classList = [] ; fullText = []
    for i in range(1,11):
        wordList = textParse(open('email/spam/spam%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(20));testSet = []
    for i in range(5):
        #随机构建训练集，这个过程属于留存交叉验证(hold-out cross validation)
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainCLasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainCLasses.append(classList[docIndex])
    p0V , p1V , pSpam = trainNBO(array(trainMat),array(trainCLasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList , docList[docIndex])
        if(classifyNB(array(wordVector) , p0V , p1V , pSpam) != classList[docIndex]):
            errorCount += 1
    print("the error rate is : " , float(errorCount)/len(testSet))

regEx = re.compile('\\W+')
mySent = 'This is the best book on Python or M.I. I have ever laid eyes upon'
listOfTokens = regEx.split(mySent)
spamTest()
