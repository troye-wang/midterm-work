from numpy import *
import matplotlib.pyplot as plt    #导入包

#定义函数
def loadDataSet():
    # 生成空列表
    dataMat = []; labelMat = []
    # fr定义为open函数打开文档
    fr = open('testSet.txt')
    # 遍历文档所有行
    for line in fr.readlines():
        # strip()删除文档结尾换行符，split()分割"\n","\t", " "
        lineArr = line.strip().split()
        #append()在列表末尾添加数据。dataMat列表：第一个数字为1.0，后面的两个数字分别是列表的每行第一个数字和第二个的数字
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # labelMat列表：添加整型数据lineArr每行第三个数
        labelMat.append(int(lineArr[2]))
    # 返回函数
    return dataMat,labelMat

#sigmod函数返回
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
'''
梯度上升算法
'''
#根据训练好的权重theta绘图
def plotBestFit(theta):
    dataMat,labelMat = loadDataSet()       #导入数据和标签
    dataArr =array(dataMat)                #生成dataMat数组
    n = shape(dataArr)[0]                  #数据个数
    xcord1 = []                            #存放label为1的点
    ycord1 = []
    xcord2 = []                            #存放label为0的点
    ycord2 = []
    #将数据按真实标签进行分类
    for i in range(n):
        #如果标签为1存放至cord1，否则就存放至cord2
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])

    #绘图
    fig = plt.figure()
    #初始位置第一行第一列第一个
    ax = fig.add_subplot(1,1,1)
    #scatter函数(x坐标,y坐标,s=点的大小,c=点的颜色，marker=点的形状s表示正方形，默认为圆。
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker='s')
    ax.scatter(xcord2, ycord2, s = 30, c = 'blue')
    #生成x的取值 -3.0——3.0,增量为0.1
    x = arange(-3.0, 3.0, 0.1)
    #y = Θ0+Θ1x1+Θ2x2
    #y=x2
    #确定分割线函数
    y = (-theta[0] - theta[1] * x) / theta[2]
    # 生成分割线
    ax.plot(x, y.T)
    #横坐标标题为X1，纵坐标标题X2
    plt.xlabel('X1')
    plt.ylabel('X2')
    #显示图片
    plt.show()


#定义函数
def gradAscent(dataMatIn, classLabels):
    #生成举证
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    #生成矩阵并转置
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    #得到矩阵大小
    m,n = shape(dataMatrix)
    #设置目标移动步长
    alpha = 0.001
    #重复次数
    maxCycles = 500
    #n行一列的单位矩阵
    weights = ones((n,1))
    '''maxCycles内的值依次赋给k
    h为sigmoid函数'''
    for k in range(maxCycles):                          #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)                 #matrix mult
        error = (labelMat - h)                          #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights



dataMat,labelMat=loadDataSet()
dataArr = array(dataMat)
theta = gradAscent(dataArr,labelMat)
print (theta)
plotBestFit(theta)