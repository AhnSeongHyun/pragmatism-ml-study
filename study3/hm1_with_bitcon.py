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

ori_x_data = xy[:, [1]].ravel()
ori_y_data = xy[:, [0]].ravel()

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

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

# TEST
answer = sess.run(hypothesis, feed_dict={X: ori_y_data[1]})
print('When X=%s, hypothesis = %s' % (ori_y_data[1], str(answer)))


# Show the linear regression result
plt.figure(1)
plt.title('Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
# 주어진 데이터들을 점으로 표시
plt.plot(x_data, y_data, 'ro')
# 예측한 일차함수를 직선으로 표시
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), 'b')
#  계산 값
plt.plot([y_data[1]], answer, 'go')
plt.show()