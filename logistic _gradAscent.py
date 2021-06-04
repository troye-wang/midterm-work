from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():  #定义函数没有接收参数
    dataMat = []; labelMat = [] #定义两个列表
    fr = open('testSet.txt')    #打开文件：
    for line in fr.readlines(): #遍历文件行
        lineArr = line.strip().split()
        #strip()删除文档结尾换行符，split()分割"\n","\t", " "
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        #添加列表属性，第一列1.0（方便计算？），浮点数第二列为文档每行第一个数据，第三列同理
        labelMat.append(int(lineArr[2]))
        #列表设置为整型原文档第二列数据
    return dataMat,labelMat
    #返回列表

def sigmoid(inX):   #sigmod函数
    return 1.0/(1+exp(-inX))

#根据训练好的权重theta绘图
def plotBestFit(theta):
    dataMat,labelMat = loadDataSet()
    dataArr =array(dataMat)
    n = shape(dataArr)[0]                  #查询矩阵维数
    xcord1 = []                            #存放label为1的点
    ycord1 = []
    xcord2 = []                            #存放label为0的点
    ycord2 = []
    #将数据按真实标签进行分类
    for i in range(n): #遍历n
        if int(labelMat[i])== 1:
            #如果标签为1存放至label1，为0则存放至label2
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    #绘图
    ax = fig.add_subplot(1,1,1)
    #定义初始位置
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker='s')
    ax.scatter(xcord2, ycord2, s = 30, c = 'blue')
    # scatter函数(x坐标,y坐标,s=点的大小,c=点的颜色，marker=点的形状s表示正方形，默认为圆。

    #生成x的取值 -3.0——3.0,增量为0.1
    x = arange(-3.0, 3.0, 0.1)
    #y = Θ0+Θ1x1+Θ2x2
    #y=x2
    y = (-theta[0] - theta[1] * x) / theta[2]
    #分割线函数
    ax.plot(x, y.T)
    #生成分割线
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    #矩阵dataMatrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    #转置矩阵labelMat
    m,n = shape(dataMatrix) #获取矩阵维数
    alpha = 0.001   #步长
    maxCycles = 500 #重复次数
    weights = ones((n,1)) #权重
    for k in range(maxCycles):                          #heavy on matrix operations
        #k遍历
        h = sigmoid(dataMatrix*weights)                 #matrix mult
        error = (labelMat - h)                          #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights



dataMat,labelMat=loadDataSet()
dataArr = array(dataMat)
theta = gradAscent(dataArr,labelMat)
print (theta)
plotBestFit(theta)