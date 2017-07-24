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

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: trainX, Y: trainY}), sess.run(W), sess.run(b))

        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        correct_prediction = tf.equal(tf.argmax(test_predict, 1), tf.argmax(testY,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        accuracy_summary = tf.summary.scalar("accuracy", accuracy)

# merge
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log", sess.graph)