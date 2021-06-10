from numpy import *


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],               #[狗感冒，请帮助]
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],           #[别带他去公园，真蠢]
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],              #[我的斑点狗真可爱，我爱它] 
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],                    #[停止发愚蠢无用的垃圾帖]
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],        #[狗吃我的牛排，如何阻止它]
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]                 #[停止买差的狗粮，真蠢]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#产生所有文档不重复的词汇集合
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#把样本文档转换为对应的词向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#训练朴素贝叶斯分类器
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#利用已经训练好的朴素贝叶斯分类器对文档分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#测试分类新文档
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()