from math import log

#创建样本数据集
def createDataSet():
    dataSet = [      [1, 1, 1, 1, 'holography'],
                     [1, 0, 0, 0, 'no'],
                     [1, 0, 1, 1, 'no'],
                     [0, 1, 1, 1, 'no'],
                     [0, 1, 0, 1, 'no'],
                     [1, 0, 0, 1, 'no'],
                     [0, 1, 1, 1, 'no'],
                     [1, 1, 0, 0, 'no'],
                     [1, 1, 1, 0, 'no']
              ]
    #样本矩阵
    labels = ['相位', '振幅', '波长', '立体视觉' ]
    #标签
    return dataSet, labels

#计算数据集的信息熵
def calcShannonEnt(dataSet): #(括号内有参数后续调用可改）
    numEntries = len(dataSet)   #数据个数
    labelCounts = {}   #字典记录每一类标签数量
    for featVec in dataSet:                                                        #遍历数据集中每个样本
        currentLabel = featVec[-1]                                                  #样本的最后一列，也就是分类结果
        if currentLabel not in labelCounts.keys():                                  #若currentLabel不再字典关键字中：
            labelCounts[currentLabel] = 0                                           #第1次出现次数为0，因为后面马上有labelCounts[currentLabel] += 1，所以没错
        labelCounts[currentLabel] += 1                                              #累计不同分类结果出现的次数，比如yes是2次，no是3次：
    shannonEnt = 0.0
    for key in labelCounts:                                                        #计算信息熵
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt       #返回信息熵

#用某个属性和该属性的取值划分数据集，划分得到的数据集不包括该属性
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        #print('a=',featVec,value,axis)
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                                       #取0到axis个元素，但不包括第axis个
            #print("b=",reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1:])                               #取从第axis+1个元素到最后一个元素
            retDataSet.append(reducedFeatVec)
            #print("c=",retDataSet )

    return retDataSet

#选择信息增益最大的属性，返回信息增益最大的属性下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1       #取属性列
    baseEntropy = calcShannonEnt(dataSet)                                          #计算基本信息熵
    bestInfoGain = 0.0;                                                            #信息增益
    bestFeature = -1                                                               #信息增益最大的属性的下标
    for i in range(numFeatures):                                                   #遍历每个属性，找信息增益最大的属性
        featList = [example[i] for example in dataSet]                             #取dataset的第i列组成列表
        uniqueVals = set(featList)                                                 #列表转换为集合，去掉重复属性值，得到唯一的属性值集合                                                                                                                                                                #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:                                                   #求某个属性的所有值划分出的数据集的信息熵，并且累加起来
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy                                        #计算每一项属性的信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature



#递归的生成决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]                                                  #得到dataset列数据
    if classList.count(classList[0]) == len(classList):                                               #第一个元素的计数，比如 yes 的个数，已经等于整个列表的长度，说明整个数据集都是一类了
        return classList[0]
    bestFeat = chooseBestFeatureToSplit(dataSet)                                                      #找最好分类属性
    bestFeatLabel = labels[bestFeat]                                                                  #得到属性名，比如flippers
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])                                                                             #删掉用过的属性名
    featValues = [example[bestFeat] for example in dataSet]                                           #得到最好的分类属性的所有属性值构成的列表
    uniqueVals = set(featValues)                                                                       #得到属性值集合，去除了重复值
    for value in uniqueVals:                                                                          #用每个属性值划分数据集
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)   #递归求解
    return myTree

dataSet,labels=createDataSet()
#print(dataDet)
#print(labels)
dTree=createTree(dataSet,labels)
print(dTree)


