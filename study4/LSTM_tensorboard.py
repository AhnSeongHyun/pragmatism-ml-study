# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 30
data_dim = 8
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 100

# Open, High, Low, Volume, Close
xy = np.genfromtxt('data/korbit_btckrw.csv', delimiter=',', dtype='float')[1:][:, [4,5,6,7,8,9,10,11]]
ori_x = xy
ori_y = xy[:, [0]]  # Close as label

xy = MinMaxScaler(xy)
x = xy
y = xy[:, [0]]  # Close as label

# build a datasetv
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

print('trainX : %s  trainY : %s' % (str(len(trainX)), str(len(trainY))))
print('testX : %s  testY : %s' % (str(len(testX)), str(len(testY))))

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
activation_func = tf.nn.relu6
cell_class = tf.contrib.rnn.BasicLSTMCell
cell = cell_class(num_units=hidden_dim, state_is_tuple=True, activation=activation_func)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim)

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares

# optimizer
optimizer_activation_func = tf.train.AdamOptimizer
optimizer = optimizer_activation_func(learning_rate)
train = optimizer.minimize(loss)

tf.summary.scalar("loss", loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("LSTM_tensorboard/%s_%s_%s_%s" % (optimizer_activation_func.__name__,
                                                                     cell_class.__name__,
                                                                     activation_func.__name__,
                                                                     str(learning_rate)), sess.graph)

    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

        summary = sess.run(merged, feed_dict={X: testX, Y: testY})
        writer.add_summary(summary, i)

        # Test step
        # test_predict = sess.run(Y_pred, feed_dict={X: testX})
        # rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        # print("RMSE: {}".format(rmse_val))