#coding:utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,6.283,0.01)

print x.shape
y=np.sin(x)


noise=np.random.random(629)*0.1
noise_y=y+noise

plt.plot(x,noise_y)

x=x.reshape(629,1)
x_train=np.ones(629)
x_train=x_train.reshape(629,1)


for i in range(1,10):
    x_train=np.hstack((x_train,x**i))

print x_train.shape




#开始训练

xx=tf.constant(x_train,dtype=tf.float32,shape=[629,10])
yy=tf.constant(noise_y,dtype=tf.float32,shape=[629,1])




xx=tf.nn.l2_normalize(xx,1)

theta=tf.Variable(tf.random_uniform([10,1], -1.0, 1.0))
bais=tf.Variable(tf.zeros([1]))

result=tf.matmul(xx,theta)+bais


cost=tf.reduce_mean(tf.square(yy-result))

optimizer = tf.train
train_step= optimizer.minimize(cost)



init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print "========="
    print sess.run(cost)
    print "========="
    for i in range(0,3000):
        sess.run(train_step)
        print sess.run(cost)
        if i%1000==0:
            print sess.run(cost)
    res=sess.run(result)
    print res
    print res.shape
    print x.shape
    plt.plot(x,res)
    plt.show()
