# -*- coding: utf-8 -*-

# Load pickled data
import pickle
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import tensorflow as tf


### Replace each question mark with the appropriate value.
import numpy as np

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline
plt.hist(y_train, bins=n_classes)
plt.hist(y_test, bins=n_classes)

import random
num_pics = 8
fig = plt.figure(figsize=(18, 18))
for i in range(num_pics):
    pic = random.randint(0, n_train-1)
    a = fig.add_subplot(1,num_pics,i+1)
    imgplot = plt.imshow(X_train[pic])
    a.set_title(pic)










EPOCHS = 20
BATCH_SIZE = 128


from tensorflow.contrib.layers import flatten

def LeNet(x, keep_prob):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # SOLUTION: Layer 1_5: Convolutional. Input = 28x28x6. Output = 20x20x6.
    conv1_5_W = tf.Variable(tf.truncated_normal(shape=(9, 9, 6, 6), mean = mu, stddev = sigma))
    conv1_5_b = tf.Variable(tf.zeros(6))
    conv1_5   = tf.nn.conv2d(conv1, conv1_5_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_5_b

    # SOLUTION: Activation.
    conv1_5 = tf.nn.relu(conv1_5)
    conv1_5 = tf.nn.dropout(conv1_5, keep_prob)    
    
    # SOLUTION: Pooling. Input = 20x20x6. Output = 14x14x6.
    conv1_5 = tf.nn.avg_pool(conv1_5, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1_5, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)


rate = 0.001

logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


save_file = 'train_model.ckpt'
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.9})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    saver.save(sess, save_file)
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


#How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)

from matplotlib.image import imread
import os
new_pics = []
for file in os.listdir('new_pics'):
    new_pic = imread('new_pics/' + file, 3).copy()
    new_pics.append(new_pic)
new_pics_array = np.array(new_pics)

def plot_in_row(images):
    fig = plt.figure(figsize=(18, 18))
    for i in range(len(images)):
        a = fig.add_subplot(1,10,i+1)
        imgplot = plt.imshow(images[i])
        a.set_title(i+1)
plot_in_row(new_pics_array)

### Run the predictions here.
### Feel free to use as many code cells as needed.
prediction_operation = tf.argmax(logits, 1)
def predict(X):
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predictions = sess.run(prediction_operation, feed_dict={x: X, keep_prob: 1.0})
        return predictions
new_pics_preds = predict(new_pics_array)

print(new_pics_preds)
for pred in new_pics_preds:
    print(sign_names.iloc[pred].SignName)


top_5_softmax_op = tf.nn.top_k(tf.nn.softmax(logits),k=5)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    top_5 = sess.run(top_5_softmax_op, feed_dict={x: new_pics_array, keep_prob: 1.0})
print(top_5)