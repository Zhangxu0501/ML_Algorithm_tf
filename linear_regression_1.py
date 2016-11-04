#coding:utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


x=np.arange(0,10,0.1)
print x.shape
y=x*0.3+3


#开始训练

#place_holder
xx=tf.constant(x,dtype=tf.float32,shape=[100,1])
yy=tf.constant(y,dtype=tf.float32,shape=[100,1])

#正规化
xx_normal=tf.nn.l2_normalize(xx,0)

#参数和偏差的定义
theta=tf.Variable(tf.random_normal(shape=[1,1],mean=1))
bais=tf.Variable(tf.random_normal([1,1]))

#cost的计算
result=tf.matmul(xx,theta)+bais
cost=tf.reduce_mean(tf.square(yy-result))

#train_step的生成
optimizer = tf.train.GradientDescentOptimizer(0.02)
train_step= optimizer.minimize(cost)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print "=================="
    print "初始代价为:"+str(sess.run(cost))
    print "=================="
    print "迭代代价"
    for i in range(0,1000):
        sess.run(train_step)
        if i%200==0:
            print sess.run(cost)
    print "=================="
    print "theta="+str(sess.run(theta))
    print "bais="+str(sess.run(bais))
    print "xx_normal="+str(sess.run(xx_normal))
