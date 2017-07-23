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


xy = np.genfromtxt('data/korbit_btckrw.csv', delimiter=',', dtype='float')[1:][:, [4,5]]

xy = MinMaxScaler(xy)
x_data = xy[:, [1]].ravel()
y_data = xy[:, [0]].ravel()




W = tf.Variable(tf.random_uniform([1]))
b = tf.Variable(tf.random_uniform([1]))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction \
    = tf.equal(tf.argmax(y_data, 1), tf.argmax(hypothesis, 1))
accuracy = tf.reduce_mean(tf.case(correct_prediction, 'float'))
accuracy_summary = tf.scalar_summary('accuracy', accuracy)

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("./log/hm3", sess.graph_def)
#http://dsmoon.tistory.com/entry/TensorBoard-Visualizing-Learning

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

