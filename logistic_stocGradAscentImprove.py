from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#根据训练好的theta绘图
def plotBestFit(theta):
    dataMat,labelMat = loadDataSet()
    dataArr =array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    #将数据按真实标签进行点分类
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker='s')
    ax.scatter(xcord2, ycord2, s = 30, c = 'blue')
    #生成x的取值 -3.0——3.0,增量为0.1
    x = arange(-3.0, 3.0, 0.1)
    #y = Θ0+Θ1x1+Θ2x2
    #y=x2
    y = (-theta[0] - theta[1] * x) / theta[2]
    ax.plot(x, y.T)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#分界线函数
def stocGradAscentImprove(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex =list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001  #虽迭代的次数增加逐渐降低                            #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))  #随机生成0到dataIndex的数字     #go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

dataMat,labelMat=loadDataSet()
dataArr = array(dataMat)
theta = stocGradAscentImprove(dataArr,labelMat,10000)
plotBestFit(theta)