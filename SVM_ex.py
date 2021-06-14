from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

#样本
x=[[1,8],[3,20],[1,15],[3,35],[5,35],[4,40],[7,80],[6,49]]
y=[1,1,-1,-1,1,-1,-1,1]

#画点
x_array= np.array(x)
x1=x_array[:, 0] #遍历样本x的x值
x2=x_array[:, 1] #遍历y值
y_array=np.array(y) #生成y数组，
i=0
for i in range(0,len(x1)) :
    # print(i) i遍历x1
    if  y_array[i] == 1:#若数组第i行为1
        plt.scatter(x1[i],x2[i],c='r',marker='*')
    elif  y_array[i] ==-1 :
        plt.scatter(x1[i], x2[i], c='b', marker='+')
        #画点
#plt.show()

#3）开始训练
clf=svm.SVC(kernel='linear')
clf.fit(x,y)

#4）预测
print("预测...")
res=clf.predict([[2,2]]); #对数据预测返回的是一个大小为n的一维数组，一维数组中的第i个值为模型预测第i个预测样本的标签；
print (res)
#绘制预测点
if res==1:
    plt.scatter(2, 2, c='r', s=120, marker='*')
elif res == -1:
    plt.scatter(2, 2, c='b', s=120,marker='+')
#plt.show()

#5）画超平面
w = clf.coef_[0]#𝜃[0]
a = -w[0] / w[1]
xx = np.linspace(-3, 10)#等差数列
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy)
#plt.show();

#6）分类器预测原始数据并画图
for i in x:
    print(np.array(i))
    print(np.array(i).reshape(1, -1))#矩阵形成一行
    res=clf.predict(np.array(i).reshape(1, -1))#预测x
    #print(res)
    if res > 0:
        plt.scatter(i[0],i[1],c='r',marker='*')
    else :
        plt.scatter(i[0],i[1],c='g',marker='+')
#plt.show();

#7）预测随机生成新数据并画图
rdm_arr=np.random.randint(1, 15, size=(15,2))   #生成15行2列，1-15随机整数
for i in rdm_arr:
    res=clf.predict(np.array(i).reshape(1, -1))
    if res > 0:
       plt.scatter(i[0],i[1],c='b',s=120, marker='*')
    else :
       plt.scatter(i[0],i[1],c='b',s=120,marker='+')
plt.show()

