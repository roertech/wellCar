# -*- coding: utf-8 -*-
# Load pickled data
import pickle
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
from utils import *
#import tensorflow as tf
training_file = 'train.p'
testing_file = 'test.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_train, y_train = train['images'], train['labels']
X_test, y_test = test['images'], test['labels']
sign_names = pd.read_csv('signnames.csv')




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
import numpy as np
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.

#plt.hist(y_train, bins=n_classes)
#plt.hist(y_test, bins=n_classes)



import random
num_pics = 8
fig = plt.figure(figsize=(18, 18))
for i in range(num_pics):
    pic = random.randint(0, n_train-1)
    a = fig.add_subplot(1,num_pics,i+1)
    imgplot = plt.imshow(X_train[pic])
    a.set_title(pic)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
print('Updated Image Shape: {}'.format(X_train[0].shape))
#print(X_train.shape)


size = 32
def apply_transform(transform, X, y):
    new_X = []
    new_y = []
    for i in range(len(X)):
        new_X.append(transform(X[i]))
        new_y.append(y[i])
    new_X = np.array(new_X)
    new_y = np.array(new_y)
    return new_X, new_y

def blur(img):
    kernel = 3
    blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return blur



def mirror(img):
    mirror = cv2.flip(img, 1)
    return mirror

def gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray


def rotate(img):
    rotate_max = 15
    M = cv2.getRotationMatrix2D((size/2, size/2), random.randint(-rotate_max, rotate_max), 1)
    rotate = cv2.warpAffine(img, M, (size, size))
    return rotate

"""
def noise(img):
    noise_max = 15
    noise_mask = np.random.randint(0, noise_max, (size, size, 3), dtype='uint8')
    noise =img+noise_mask
    return noise
"""
def crop(img):
    new_X, new_y = [], []
    crop_size = 28
    crop_x, crop_y = random.randint(crop_size, size), random.randint(crop_size, size)
    crop = img[crop_y-crop_size:crop_y, crop_x-crop_size:crop_x]
    crop = cv2.resize(crop, (size, size))
    return crop

def balance_classes(X, y):
    values, counts = np.unique(y, return_counts=True)
    mode_class_count = max(counts)
    transforms = [rotate, crop]  # all of these have a random component
    for class_ in values:
        new_imgs = []
        class_imgs = X[y == class_]
        diff_count = mode_class_count - counts[values == class_][0]
        for i in range(diff_count):
            original_img = class_imgs[random.randint(0, len(class_imgs)-1)]
            transform = random.choice(transforms)
            new_imgs.append(transform(original_img))
        if new_imgs:
            X = np.concatenate((X, np.array(new_imgs)))
            y = np.concatenate((y, [class_] * diff_count))
    return X, y


X_train, y_train = balance_classes(X_train, y_train)


X_train_blur, y_train_blur = apply_transform(blur, X_train, y_train)
X_train_mirror, y_train_mirror = apply_transform(mirror, X_train, y_train)
X_train_gray, y_train_gray = apply_transform(gray, X_train, y_train)
X_train_rotate, y_train_rotate = apply_transform(rotate, X_train, y_train)
#X_train_noise, y_train_noise = apply_transform(noise, X_train, y_train)
X_train_crop, y_train_crop = apply_transform(crop, X_train, y_train)
X_train = np.concatenate((X_train, X_train_blur, X_train_mirror, X_train_gray, X_train_rotate,  X_train_crop))#X_train_noise,
y_train = np.concatenate((y_train, y_train_blur, y_train_mirror, y_train_gray, y_train_rotate,  y_train_crop))#y_train_noise,
print(X_train.shape)

X_train, y_train = shuffle(X_train, y_train)
"""
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
    """
