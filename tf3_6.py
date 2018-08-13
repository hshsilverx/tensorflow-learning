#coding:utf-8
#步骤0（准别）：导入模块，生成模拟数据集

import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed=23455

#基于seed产生随机数
rng = np.random.RandomState(seed)

#随机数返回32行2列的矩阵，表示32组，体积和重量，座位输入的数据集
X = rng.rand(32,2)

#从X这个32行2列矩阵中取出一行，判断如果和＜1给Y赋值1，否则赋值0，座位输入数据的标签（可以看做一个正确答案）
Y = [[int(x0+x1<1)]for(x0,x1)in X]
print"X:\n",X
print"Y:\n",Y


#步骤1（前向传播）：定义神经网络的输入参数和输出，定义前项传播过程
x=tf.placeholder(tf.float32, shape = (None,2))
y_=tf.placeholder(tf.float32, shape = (None,1))

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)


#步骤2（反向传播）：定义损失函数、反向传播方法(均方误差)
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#还可以用如下两种方法，选一就行
#train_step = tf.train.MomentumOptimizer(0.001,0,9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#步骤3（会话）：生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    #输出目前未经训练的参数取值
    print "w1:\n",sess.run(w1)
    print "w2:\n",sess.run(w2)
    print "\n"
    #训练模型
    STEPS =3000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%32
        end=start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i %500==0:
            total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d training step(s), loss on all data is %g" % (i,total_loss))
    #输出训练后的参数值
    print"\n"
    print "w1:\n",sess.run(w1)
    print "w2:\n",sess.run(w2)

#改进：增大STEPS，修改BATCH_SIZE大小，修改反向传播的方法。

