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

def weight_variable(shape):
  initial = tf.truncated_normal(shape=[])
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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


x_data=tf.placeholder(dtype=tf.float32,shape=[None,2])
y_labels=tf.placeholder(dtype=tf.float32,shape=[None,2])

# theta=weight_variable([2,2])
# b = bias_variable([2])
theta=tf.Variable(tf.random_normal(shape=[2,2]))
b=tf.Variable(tf.random_normal(shape=[2]))


h=tf.matmul(x_data,theta)+b
h1=tf.nn.sigmoid_cross_entropy_with_logits(h, y_labels)
cost = tf.reduce_mean(h1) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost) # construct optimizer

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(h1,feed_dict={x_data:data,y_labels:labels})

    # for i in range(0,1000):
    #     sess.run(train_op,feed_dict={x_data:data,y_labels:labels})
    #     print sess.run(cost,feed_dict={x_data:data,y_labels:labels})
    # end=sess.run(h,feed_dict={x_data:data,y_labels:labels})
    #
    # result=list()
    #
    # for i in end:
    #     result.append(get_max(i))
    #
    # count=0
    # for i in range(0,200):
    #     if(result[i]==labels_r[i]):
    #         count=count+1
    # print "准确率为"+str(count/(len(result)+0.0))
    # print count











