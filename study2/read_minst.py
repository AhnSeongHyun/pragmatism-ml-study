# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


for key in ['train', 'test', 'validation']:
    print("mnist.%s :" % key)
    minst_dataset= getattr(mnist, key)
    print("\tnum_examples : %d" % minst_dataset.num_examples)
    print("\timages count : %d" % len(minst_dataset.images))
    print("\tlabels count : %d" % len(minst_dataset.labels))
    print("\timages[0] len : %d" % len(minst_dataset.images[0]))
    print("\tlabels[0] len : %d" % len(minst_dataset.labels[0]))
    print("\tlabels[0]   : %s" %  str(minst_dataset.labels[0]))

#print(mnist.train.next_batch(1))
