#coding:utf-8
import feedparser
import operator
from numpy import *
import re

#input is big string, #output is word list
def textParse(bigString):
    listOfTokens = re.split(r'\W+' , bigString)
    return [tok.lower() for tok in listOfTokens if (len(tok) > 1)]

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

#朴素贝叶斯分类函数
def classifyNB(vec2Classify , p0Vec , p1Vec , pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1) #相当于是后验概率：概率密度乘以先验概率
    p0 = sum(vec2Classify*p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#词袋模型中的函数
def bagOfWords2Vec(vocabList , inputSet):
    #函数的输入参数为词汇表及某个文档，输出文档向量，向量每一个元素为0或者1，表示词汇表中的单词在输入文档中是否出现
    #创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        #else:print("the word : %s is not in my Vocabulary !" % word)
    return returnVec

def calcMostFreq(vocabList , fullText):
    #计算出现频率
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items() , key = operator.itemgetter(1) , reverse=True)
    return sortedFreq[:30]

def localWords(feed1 , feed0):
    docList = [] ; classList = [] ; fullText  = []
    minLen = min(len(feed1['entries']) , len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList , fullText)
    #去掉出现频率最高的30个词
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(minLen)) ; testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [] ; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList , docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V , p1V , pSpam = trainNBO(array(trainMat) , array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList , docList[docIndex])
        if classifyNB(array(wordVector) , p0V , p1V , pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is : ' , float(errorCount)/len(testSet))
    return vocabList , p0V , p1V

def getTopWords(ny , sf):
    vocabList , p0V , p1V = localWords(ny , sf)
    topNY = [] ; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -4.5: topSF.append((vocabList[i] , p0V[i]))
        if p1V[i] > -4.5: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF , key = lambda pair:pair[1] , reverse=True)
    print("SF**SF**SF**SF**SF**SF*SF**SF**SF**SF")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY")
    for item in sortedNY:
        print(item[0])


if __name__ == "__main__":
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny , sf)