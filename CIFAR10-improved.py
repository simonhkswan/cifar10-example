from __future__ import print_function
import shutil
import os.path
import numpy as np
import tensorflow as tf
import zipfile
import time
import urllib
import gzip
import tarfile
import sys
import random

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_DIR = '../data/'
EXPORT_DIR = './graphs'

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 3072  # CIFAR data input (img shape: 32*32)
n_classes = 10  # CIFAR total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name = 'x')
y = tf.placeholder(tf.float32, [None, n_classes], name = 'y')
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(DATA_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(DATA_DIR, 'cifar-10-batches-py')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(DATA_DIR)

maybe_download_and_extract()

def unpickle(file):
    import pickle
    with open(file, 'rb') as infile:
    	dicti = pickle.load(infile, encoding='bytes')
    return dicti

cifar10 = unpickle("../data/cifar-10-batches-py/data_batch_1")

def generate_batch(dataset, batch_size):
	# returns a batch of images and labels from a set of data.
	# dataset is a dictionary with b'data' and b'labels' keys.
	images = dataset[b'data']
	labels = dataset[b'labels']
	i = random.randint(0,images.shape[0]-batch_size)
	y = np.zeros((batch_size, n_classes))
	for j in range(y.shape[0]):
		y[j][labels[i+j]]=1
	return images[i:i+batch_size], y

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, c=1):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, c], strides=[1, k, k, 1],
                          padding='SAME')


# Create Model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 32, 32, 3])

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
    # 5x5 conv, 3 input channels (r,g,b), 64 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 64]), name='wc1'),
    # 5x5 conv, 64 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 64]), name='wc2'),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8 * 8 * 64, 1024]), name='wd1'),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]), name='out')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='bout')
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

test_data = unpickle("../data/cifar-10-batches-py/test_batch")
test_imgs, test_labels = generate_batch(test_data, 500)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = generate_batch(cifar10, batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_imgs[:256], y: test_labels[:256], keep_prob: 1.}))
    WC1 = weights['wc1'].eval(sess)
    BC1 = biases['bc1'].eval(sess)
    WC2 = weights['wc2'].eval(sess)
    BC2 = biases['bc2'].eval(sess)
    WD1 = weights['wd1'].eval(sess)
    BD1 = biases['bd1'].eval(sess)
    W_OUT = weights['out'].eval(sess)
    B_OUT = biases['out'].eval(sess)

# Create new graph for exporting
summary = tf.summary.FileWriter('./graphs')

g = tf.Graph()
with g.as_default():
    x_2 = tf.placeholder("float", shape=[None, 3072], name="input")

    WC1 = tf.constant(WC1, name="WC1")
    BC1 = tf.constant(BC1, name="BC1")
    x_image = tf.reshape(x_2, [-1, 32, 32, 3])
    CONV1 = conv2d(x_image, WC1, BC1)
    MAXPOOL1 = maxpool2d(CONV1, k=2)

    WC2 = tf.constant(WC2, name="WC2")
    BC2 = tf.constant(BC2, name="BC2")
    CONV2 = conv2d(MAXPOOL1, WC2, BC2)
    MAXPOOL2 = maxpool2d(CONV2, k=2)

    WD1 = tf.constant(WD1, name="WD1")
    BD1 = tf.constant(BD1, name="BD1")

    FC1 = tf.reshape(MAXPOOL2, [-1, WD1.get_shape().as_list()[0]])
    FC1 = tf.add(tf.matmul(FC1, WD1), BD1)
    FC1 = tf.nn.relu(FC1)

    W_OUT = tf.constant(W_OUT, name="W_OUT")
    B_OUT = tf.constant(B_OUT, name="B_OUT")

    # skipped dropout for exported graph as there is no need for already calculated weights

    OUTPUT = tf.nn.softmax(tf.matmul(FC1, W_OUT) + B_OUT, name="output")

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    graph_def = g.as_graph_def()
    summary.add_graph(g)
    tf.train.write_graph(graph_def, EXPORT_DIR, 'cifar_model.pb', as_text=False)

    # Test trained model
    y_train = tf.placeholder("float", [None, 10])
    correct_prediction = tf.equal(tf.argmax(OUTPUT, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("check accuracy %g" % accuracy.eval(
            {x_2: test_imgs, y_train: test_labels}, sess))
