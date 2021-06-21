import tensorflow as tf #载入tensorflow
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #import ERROR + FATAL hide waring

mnist=tf.keras.datasets.mnist   #准备数据集归一化（减小数值差异）
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#堆叠搭建模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),    #28*28像素格
  tf.keras.layers.Dense(128, activation='relu'),    #实现了这个操作:output = activation(dot(乘积)(input, kernel) + bias);128维度，激活relu函数
  tf.keras.layers.Dropout(0.2),  #dropout 是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率（rate默认0.5)将其暂时从网络中丢弃
  tf.keras.layers.Dense(10, activation='softmax')   #同上上
])

model.compile(optimizer='adam', #调用配置模型的优化器
              loss='sparse_categorical_crossentropy',   #调用配置模型的损失函数
              metrics=['accuracy']) #调用配置模型评价的方法

model.fit(x_train, y_train, epochs=5)   #训练数据（数据x,y值，epochs=迭代次数）

model.evaluate(x_test,  y_test, verbose=1) #评估已经训练过的模型.返回数据和误差，verbose日志显示

plt.imshow(x_train[0], cmap='gray')

