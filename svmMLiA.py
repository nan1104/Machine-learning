#coding:utf-8、
#SMO算法，用于训练SVM。SMO表示序列最小化（Sequential Minimal Optimization）
from numpy import *

def loadDataSet(fileName):
    dataMat = [] ; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split() #split('\t')会报错
        dataMat.append([float(lineArr[0]) , float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat , labelMat

#辅助函数，用于在某个区间范围内随机选择一个整数
def selectJrand(i , m):
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
    return j

#辅助函数，用于在数值太大时，对其进行调整
def clipAlpha(aj , H , L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn , classLabels , C , toler , maxIter):
    #五个输入参数：数据集、类别标签、常数C、容错率、和推出前最大的循环次数
    dataMatrix = mat(dataMatIn) ; labelMat = mat(classLabels).transpose()  #转置了类别标签，得到一个列向量
    b = 0 ; m,n = shape(dataMatrix) #m代表样本数，n代表特征数
    alphas = mat(zeros((m,1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChagnged = 0 #该变量用于记录alpha是否已经进行优化d
        for i in range(m):
            fXi = float(multiply(alphas , labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #如果alpha可以更改进入优化过程
                j = selectJrand(i , m) #随机选择一个第二个alpha
                fXj = float(multiply(alphas , labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0 , alphas[j] - alphas[i])
                    H = min(C , C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H :
                    print("L==H");continue
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0 : print("eta>=0") ; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j] , H , L)
                if(abs(alphas[j] - alphaJold) < 0.00001) : print("j not moving enough");continue
                alphas[i] += labelMat[j] * labelMat[i]*(alphaJold - alphas[j]) #对i进行修改，修改量与j相同，但方向相反
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j] - alphaJold) * dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j] - alphaJold) * dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]) : b= b1
                elif (0 < alphas[j]) and (C > alphas[j]) : b= b2
                else : b = (b1 + b2)/2.0  #设置常数项
                alphaPairsChagnged += 1
                print("iter : %d i = %d, pairs changed %d" % (iter , i,alphaPairsChagnged))
        if(alphaPairsChagnged == 0) : iter += 1
        else : iter = 0
        print("iteration number : %d" % iter)
    return b , alphas

if __name__ == "__main__":
    dataArr , labelArr = loadDataSet('testSet.txt')
    b , alphas = smoSimple(dataArr , labelArr , 0.6 , 0.001 , 40)
    print(b)
