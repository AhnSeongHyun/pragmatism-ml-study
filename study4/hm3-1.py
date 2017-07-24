from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

with tf.name_scope("input") as scope:
    x = tf.placeholder(tf.float32, [None, 784])

with tf.name_scope("weight") as scope:
    W = tf.Variable(tf.zeros([784, 10]))

with tf.name_scope("bias") as scope:
    b = tf.Variable(tf.zeros([10]))

with tf.name_scope("layer1") as scope:
    y = tf.nn.softmax(tf.matmul(x, W) + b)

w_hist = tf.summary.histogram("weight", W)
b_hist = tf.summary.histogram("bias", b)
y_hist = tf.summary.histogram("y", y)
epoch = 2000
with tf.name_scope("y_") as scope:
    y_ = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("cost") as scope:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cost_sum = tf.summary.scalar("cost", cross_entropy)

learning_curve = 0.8
ac_func = 'GradientDescentOptimizer'
with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(learning_curve).minimize(cross_entropy)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("board/mnist_%s_%s_%s" % (ac_func, str(learning_curve), str(epoch)), sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(epoch):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 10 == 0:
        summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(summary, i)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



