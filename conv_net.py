from __future__ import print_function

import os
import cv2
import numpy as np
import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters

learning_rate = 0.00001
training_iters = 200000
batch_size = 32
display_step = 10

# Network Parameters
n_input = 54000 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def get_next_batch(folder, index, batch_size):
    train_input = []
    train_output = []
    ix = 0
    for root, dirs, files in os.walk(folder+'/negative'):
        for basename in files:
            if ix > index and ix <= index + batch_size:
                filename = os.path.join(root, basename)
                im = cv2.imread(filename)
                vec = im.flatten()
                train_input.append(vec)
                train_output.append(np.zeros(10))
            ix += 1

    ix = 0
    for root, dirs, files in os.walk(folder+'/positive'):
        for basename in files:
            if ix > index and ix <= index + batch_size:
                filename = os.path.join(root, basename)
                spl = basename.replace('.png','').split("_")

                if "flipped" in spl:
                    idx = 5
                else:
                    idx = 4

                out_vec = np.zeros(10)
                for j in xrange(len(spl[idx:])):
                    num = int(spl[idx+j])
                    if j % 2 == 0:
                        norm_term = 180
                    else:
                        norm_term = 100
                    out_vec[j] = (num*1.0/norm_term)

                im = cv2.imread(filename)
                vec = im.flatten()
                train_input.append(vec)
                train_output.append(out_vec)
            ix += 1

    train_input = np.array(train_input)
    train_output = np.array(train_output)

    p = np.random.permutation(len(train_output))
    train_input = train_input[p]
    train_output = train_output[p]
    return (train_input, train_output)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 180, 100, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # # Max Pooling (down-sampling)
    # conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.softmax(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # 'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([45*25*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    # 'bc3': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

#predicted_value = tf.argmax(pred,1)
is_increasing = lambda L: reduce(lambda a,b: b if a < b else 9999 , L)!=9999

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), tfargmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    err = []
    count = 0
    while step * batch_size < training_iters:
        batch_x, batch_y = get_next_batch('resized_train',step * batch_size,batch_size)

        if(len(batch_x) == 0 ):
            step = 1

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, val = sess.run([cost,pred], feed_dict= { x: batch_x,y: batch_y,keep_prob: 1.})
            err.append(loss)

            if len(err) > 3 and is_increasing(err[-3:]) and count > 2:
                break
            elif len(err) > 3 and is_increasing(err[-3:]):
                count+=1
                print(count)
            #val = predicted_value.eval(feed_dict= { x: batch_x,y: batch_y,keep_prob: 1.})
            print(val)
            print(batch_y)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +"{:.6f}".format(loss))
        step += 1
    print("Optimization Finished!")
    pd = sess.run(pred, feed_dict = { x: batch_x,keep_prob: dropout})
    print(pd)
    print(batch_y)

    # Calculate accuracy for 256 mnist test images
    # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],y: mnist.test.labels[:256],keep_prob: 1.}))
