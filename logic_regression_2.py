#coding:utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


x0=np.random.randint(0,5,100)
x1=np.random.randint(10,15,100)

y0=np.random.randint(0,10,100)
y1=np.random.randint(10,20,100)

x=np.hstack((x0,x1))
x=x+np.random.rand(200)
y=np.hstack((y0,y1))+np.random.rand(200)
labels=list()

for i in range(0,200):
    if i<=99:
        labels.append([0,1])
    else:
        labels.append([1,0])
labels_r=list()

for i in range(0,200):
    if i<=99:
        labels_r.append(1)
    else:
        labels_r.append(0)

# plt.scatter(x[0:100],y[0:100],s=20,c="red")
# plt.scatter(x[100:],y[100:],s=20,c="blue")
# plt.show()



def get_max(arr):
    max=arr[0]
    label=0
    for i in range(0,len(arr)):
        if arr[i]>max:
            max=arr[i]
            label=i
    return label

data=list()
for i in range(0,200):
    data.append([x[i],y[i]])

data=np.float32(data)
labels=np.float32(labels)


#数据准备结束,开始训练

x_data=tf.placeholder(dtype=tf.float32,shape=[None,2])
y_labels=tf.placeholder(dtype=tf.float32,shape=[None,2])

#正规化和偏差
x_normal=tf.nn.l2_normalize(x_data,0)
theta=tf.Variable(tf.random_normal(shape=[2,2]))
b=tf.Variable(tf.random_normal(shape=[2]))


#z的计算
h=tf.matmul(x_normal,theta)+b

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(h, y_labels)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(300.0).minimize(cost) # construct optimizer

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(0,1000):
        sess.run(train_op,feed_dict={x_data:data,y_labels:labels})
        if i%200==0:
            print sess.run(cost,feed_dict={x_data:data,y_labels:labels})

    get_max_labels=sess.run(h,feed_dict={x_data:data,y_labels:labels})
    result=list()

    for i in get_max_labels:
        result.append(get_max(i))

    count=0
    for i in range(0,200):
        if(result[i]==labels_r[i]):
            count=count+1
    print "准确率为"+str(count/(len(result)+0.0))











