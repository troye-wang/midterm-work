from numpy import * #numpy导入

#加载数据
def loadDataSet(fileName):#可向函数传递信息
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))   #计算浮点数
        dataMat.append(fltLine) #扩展
    return dataMat

#计算欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

#随机产生K个质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]#第一维度矩阵长度
    #array([[第一行数组],[第二行数组]])
    centroids = mat(zeros((k, n)))#k行n列零数组
    for j in range(n):
        minJ = min(dataSet[:, j])#寻找最小值
        rangeJ = float( max(dataSet[:, j]) - minJ)#j的范围最大值减去最小值
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))#计算数组最小值+范围*随机k行1列并转化为矩阵
    return centroids

#K均值聚类算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]#读取第二维长度
    clusterAssment = mat(zeros((m,2)))#零矩阵
    centroids = createCent(dataSet, k)#随机质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1    #inf代表正无穷
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])  #计算欧氏距离
                if distJI < minDist:
                    minDist = distJI; minIndex = j  #找到质心
            if clusterAssment[i,0] != minIndex: #如果当前质心不是最小距离的质心则更新质心
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]   #返回数据样本中所有非零的质心
            centroids[cent,:] = mean(ptsInClust, axis=0)    #输出一行非零质心矩阵
    return centroids, clusterAssment

dataMat=mat(loadDataSet('testSet2.txt'))    #数据样本
mycentroids,clustAssing = kMeans(dataMat,4) #该算法

