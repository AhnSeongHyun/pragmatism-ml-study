# -*- coding:utf-8 -*-

import tensorflow as tf
import random
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
LOGDIR='./model_save'

# hyper parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 100


class Model(object):
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])  # N 개의 이미지
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])   # N개의 img 28x28x1 (black/white)
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # L1 ImgIn shape=(?, 28, 28, 1)
            # 필터의 크기 3*3, 32개의 필터
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            #    Conv     -> (?, 28, 28, 32)
            #    Pool     -> (?, 14, 14, 32)

            # 이동거리 1
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            '''
            Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
            Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
            Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
            '''

            # L2 ImgIn shape=(?, 14, 14, 32)
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            #    Conv      ->(?, 14, 14, 64)
            #    Pool      ->(?, 7, 7, 64)
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
            '''
            Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
            Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
            Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
            Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
            '''

            # Final FC 7x7x64 inputs -> 10 outputs
            W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 10], stddev=0.01))
            # W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([10]), name='b')
            self.logits = tf.matmul(L2_flat, W3) + b

        # define cost/loss & optimizer
        self.activation_func = tf.train.AdamOptimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = self.activation_func(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, y_test):
        return self.sess.run(self.logits, feed_dict={self.X : x_test, self.Y : y_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})

    def save_model(self, model_path):
        # 모델 저장
        saver = tf.train.Saver()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver.save(sess, os.path.join(model_path, "model.ckpt"), i)


sess = tf.Session()
m1 = Model(sess=sess, name='m1')


with tf.name_scope("M1") as scope:
    tf.summary.scalar("cost", m1.cost)
    tf.summary.scalar("accuracy", m1.accuracy)

sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("minst_tensorboard/mnist_%s_%s_%s" % (m1.activation_func.__name__, str(learning_rate), str(training_epochs)), sess.graph)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(1000 / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(x_data=batch_xs, y_data=batch_ys)
        avg_cost += c / total_batch

    test_xs, test_ys = mnist.validation.next_batch(batch_size)

    # tensorboard
    summary = sess.run(merged, feed_dict={m1.X: test_xs, m1.Y: test_ys})
    writer.add_summary(summary, epoch)
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))

m1.save_model(model_path=LOGDIR)

#
# with tf.name_scope("Accuracy/Cost") as scope:
#
#     # Accuracy
#     correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#     tf.summary.scalar("cost", cost)
#     tf.summary.scalar("accuracy", accuracy)
#
#
# # initialize
# sess = tf.Session()
# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter("minst_tensorboard/mnist_%s_%s_%s" % (ac_func, str(learning_rate), str(training_epochs)), sess.graph)
#
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
#
# # Get one and predict
# r = random.randint(0, mnist.test.num_examples - 1)
# print(r)
# print(tf.argmax(mnist.test.labels[r:r + 1], 1))
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
#
# saver = tf.train.Saver()
#
# print('Learning started. It takes sometime.')
# for epoch in range(training_epochs):
#     avg_cost = 0
#     total_batch = int(1000 / batch_size)
#
#     for i in range(total_batch):
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         feed_dict = {X: batch_xs, Y: batch_ys}
#         c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
#         avg_cost += c / total_batch
#
#     test_xs, test_ys = mnist.validation.next_batch(batch_size)
#     summary = sess.run(merged, feed_dict={X: test_xs, Y: test_ys})
#     writer.add_summary(summary, epoch)
#
#     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
# print('Learning Finished!')
# saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
# print('Save Model : %s' % save_file)
#
# print('Accuracy:', sess.run(accuracy, feed_dict={
#       X: mnist.test.images, Y: mnist.test.labels}))
#
#
#
