# Date: 2018-08-16 16:09
# Author: Enneng Yang
# Abstract：Truncated Gradient

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("Data/MNIST_data/", one_hot=True)

method_name = 'TG'
# training Parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 100
display_step = 1
tg_theta = 0.001
clip_value_min = -0.001
clip_value_max = 0.001

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape: 28*28=784
y = tf.placeholder(tf.float32, [None, 10])   # 0-9 digits recognition: 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
gradients = optimizer.compute_gradients(cost)
# tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)
capped_gradients = [(tf.clip_by_value(grad, clip_value_min, clip_value_max), var) for grad, var in gradients]
optimizer = optimizer.apply_gradients(capped_gradients)

# Initializing the variables
init = tf.global_variables_initializer()

all_loss = []
all_step = []

plt.title('Optimization method:'+ method_name)
plt.xlabel('training_epochs')
plt.ylabel('loss')

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):

        avg_cost = 0.
        epoch_cost = 0.

        total_batch = int(mnist.train.num_examples/batch_size)

        # Loop over all batches
        for i in range(total_batch):

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c_ = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            # Compute average loss
            epoch_cost += c_

        avg_cost = epoch_cost / total_batch

        # opt loss
        all_loss.append(avg_cost)
        all_step.append(epoch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

plt.plot(all_step, all_loss, color='red', label=method_name)
plt.legend(loc='best')

plt.show()
plt.pause(1000)




