#coding:utf-8
#改进后的SMO算法
from numpy import *

class optStruct:
    #这里的类是为了作为一个数据结构来使用对象
    def __init__(self,dataMatin , classLabels , C , toler):
        self.X = dataMatin
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatin)[0]
        self.alphas = mat(zeros((self.m ,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #eCache的第一列给出的是eCache是否有效的标志位，第二位给出的是实际的E值

#辅助函数，用于在某个区间范围内随机选择一个整数
def selectJrand(i , m):
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
    return j

def loadDataSet(fileName):
    dataMat = [] ; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split() #split('\t')会报错
        dataMat.append([float(lineArr[0]) , float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat , labelMat

def calcEk(oS , k):
    #计算Ei
    fXk = float(multiply(oS.alphas,oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    Ek = fXk - oS.labelMat[k]
    return Ek

def selectJ(i , oS , Ei):
    #用于选择第二个alpha或者说内循环的alpha值
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei] #将输入值Ei在缓存中设置成有效的，有效意味着已经计算好了
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS , k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k;maxDeltaE = deltaE;Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i , oS.m)
        Ej = calcEk(oS , j)
    return j , Ej

def updateEk(oS , k):
    Ek = calcEk(oS , k)
    oS.eCache[k] = [1,Ek]

#辅助函数，用于在数值太大时，对其进行调整
def clipAlpha(aj , H , L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def innerL(i , oS):
    Ei = calcEk(oS , i)
    if((oS.labelMat[i]*Ei < -oS.tol)and(oS.alphas[i] < oS.C) or
               (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j , Ej = selectJ(i , oS ,Ei)
        alphaIold = oS.alphas[i].copy() ; alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H");return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T \
              - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0: print("eta>=0");return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS , j) #更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough");return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # 对i进行修改，修改量与j相同，但方向相反
        updateEk(oS , i) #更新误差缓存
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] \
             * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] \
            * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0  # 设置常数项
        return 1
    else:return 0

def smoP(dataMatin , classLabels , C , toler , maxIter , kTup = ('lin' , 0)):
     oS = optStruct(mat(dataMatin) , mat(classLabels).transpose() , C ,toler)
     iter = 0
     entireSet = True; alphaPairsChanged = 0
     while(iter < maxIter) and ((alphaPairsChanged > 0)or(entireSet)):
         alphaPairsChanged = 0
         if entireSet:
             for i in range(oS.m): #先随机选取一个i值
                 alphaPairsChanged += innerL(i , oS) #通过调用innerL选择第二个alpha,并对其进行更新
             print("fullSet , iter:%d i:%d , pairs changed %d" % (iter ,i ,alphaPairsChanged))
             iter += 1
         else: #循环遍历所有的非边界alpha
             nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A <C))[0]
             for i in nonBoundIs:
                 alphaPairsChanged += innerL(i ,oS)
                 print("non-nound , iter:%d i:%d ,pairs change %d" %(iter , i, alphaPairsChanged))
             iter += 1
         if entireSet :entireSet = False
         elif (alphaPairsChanged == 0) :entireSet = True
         print("iteration number: %d" % iter)
     return oS.b , oS.alphas

def clacWs(alphas , dataArr , classLabels):
    #计算权重w
    X = mat(dataArr);labelMat = mat(classLabels).transpose()
    m , n =shape(X)
    w = zeros((n , 1))
    for i in range(m):
        #大部分alpha为0，非0alpha对应的为支持向量
        w += multiply(alphas[i]*labelMat[i] , X[i,:].T)
    return w

if __name__ == "__main__":
    dataArr , labelArr = loadDataSet("testSet.txt")
    b , alphas = smoP(dataArr , labelArr , 0.6 , 0.001 , 40)
    ws = clacWs(alphas , dataArr , labelArr)
    print(ws)
    dataMat = mat(dataArr)
    count = 0
    for i in range(100):
        if (dataMat[i]*mat(ws) + b)*labelArr[i]>0:
            count += 1
    print("分类正确率为：%f"% (count/100))

