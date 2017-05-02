'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Import theano for data_provide
import time
import logging

import numpy
import theano
import theano.tensor as T
import cPickle, gzip
import random

#Load pic pkl
print("import_another_data_set")
dataset = './flower.pkl'
f = open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y0 = shared_dataset(test_set)
valid_set_x, valid_set_y0 = shared_dataset(valid_set)
train_set_x, train_set_y0 = shared_dataset(train_set)


print(train_set_x.eval().shape)
print(train_set_y0.eval().shape)
print(valid_set_x.eval().shape)


# Parameters
learning_rate = 0.001
training_iters = 250000
batch_size = 128
# batch_size = 20
display_step = 10
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 17 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

#change_train_label
train_set_y1 = [[0 for col in range(n_classes)] for row in range(train_set_x.eval().shape[0])]
# print(len(train_set_y1))
for i in xrange(train_set_y0.eval().shape[0]):
    temp = train_set_y0.eval()[i]
    train_set_y1[i][temp] = 1
shared_y1 = theano.shared(numpy.asarray(train_set_y1, dtype=theano.config.floatX))
train_set_y = T.cast(shared_y1, 'float64')
#change_valid_label
valid_set_y1 = [[0 for col in range(n_classes)] for row in range(valid_set_x.eval().shape[0])]
# print(len(valid_set_y1))
for i in xrange(valid_set_y0.eval().shape[0]):
    temp = valid_set_y0.eval()[i]
    valid_set_y1[i][temp] = 1
shared_y1 = theano.shared(numpy.asarray(valid_set_y1, dtype=theano.config.floatX))
valid_set_y = T.cast(shared_y1, 'float64')
#change_test_label
test_set_y1 = [[0 for col in range(n_classes)] for row in range(test_set_x.eval().shape[0])]
# print(len(test_set_y1))
for i in xrange(test_set_y0.eval().shape[0]):
    temp = test_set_y0.eval()[i]
    test_set_y1[i][temp] = 1
shared_y1 = theano.shared(numpy.asarray(test_set_y1, dtype=theano.config.floatX))
test_set_y = T.cast(shared_y1, 'float64')

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

valid_num = 1

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    print("train_try")
    while step * batch_size < training_iters:
        # print("train_try")
        #my_way
        for minibatch_index in range(n_train_batches):
            
            index = minibatch_index
            index_l = index * batch_size
            index_h = (index + 1) * batch_size

            sess.run(optimizer, feed_dict={x: train_set_x.eval()[index_l: index_h], y: train_set_y.eval()[index_l: index_h],
                                           keep_prob: dropout})
            if valid_num == 10:
                sess.run(optimizer, feed_dict={x: valid_set_x.eval()[int(index_l/2): int(index_h/2)], y: valid_set_y.eval()[int(index_l/2): int(index_h/2)],
                                           keep_prob: dropout})
                valid_num = 1

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: train_set_x.eval()[index_l: index_h],
                                                                  y: train_set_y.eval()[index_l: index_h],
                                                                  keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                loss, acc = sess.run([cost, accuracy], feed_dict={x: valid_set_x.eval()[int(index_l/2): int(index_h/2)],
                                                                  y: valid_set_y.eval()[int(index_l/2): int(index_h/2)],
                                                                  keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Valid Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
            valid_num += 1

    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_set_x.eval()[:256],
                                      y: test_set_y.eval()[:256],
                                      keep_prob: 1.}))
