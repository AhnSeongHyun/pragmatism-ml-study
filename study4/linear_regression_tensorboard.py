# -*- coding:utf-8 -*-
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


xy = np.genfromtxt('data/korbit_btckrw.csv', delimiter=',', dtype='float')[1:][:, [4,9]]

xy = MinMaxScaler(xy)
x_data = xy[:, [1]].ravel()[40000:]
y_data = xy[:, [0]].ravel()[40000:]


train_size = int(len(y_data) * 0.7)
test_size = len(y_data) - train_size
trainX, testX = np.array(x_data[0:train_size]), np.array(x_data[train_size:len(x_data)])
trainY, testY = np.array(y_data[0:train_size]), np.array(y_data[train_size:len(y_data)])

print('trainX : %s  trainY : %s' % (str(len(trainX)), str(len(trainY))))
print('testX : %s  testY : %s' % (str(len(testX)), str(len(testY))))


W = tf.Variable(tf.random_uniform([1]))
b = tf.Variable(tf.random_uniform([1]))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b


epoch = 2000
learning_curve = 0.01
activation_func = tf.train.AdamOptimizer


cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = activation_func(learning_rate=learning_curve)
train = optimizer.minimize(cost)

tf.summary.scalar("cost", cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("linear_reg_tensorboard/%s_%s_%s" % (activation_func.__name__, str(learning_curve), str(epoch)), sess.graph)

for step in range(epoch):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        summary = sess.run(merged, feed_dict={X: trainX, Y: trainY})
        writer.add_summary(summary, step)