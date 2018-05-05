
# coding: utf-8
import pickle
import numpy as np
import time
from model.net import deepnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("~/roerteck/datasets/minist_data/", one_hot=True)

#X_train, y_train           = mnist.train.images, mnist.train.labels
#X_validation, y_validation = mnist.validation.images, mnist.validation.labels
#X_test, y_test             = mnist.test.images, mnist.test.labels

"""
assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))
"""

"""
import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))
"""

#image = X_train[index].squeeze()#去除1 维度

#x = tf.placeholder(tf.float32, shape=[None, 240, 320, 3], name='x')
#y_ = tf.placeholder(tf.float32, shape=[None, 3], name='y_')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
y_conv, keep_prob = deepnn(x)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#train and save model
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(30)
    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

saver = tf.train.Saver()
#model_file = os.path.dirname(os.path.realpath(__file__)) + '/' + os.path.basename(__file__)

saver.save(sess, './pretrain/model' )    
print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#feed_dict = {x: mnist.test.images[:100], y_: mnist.test.labels[:100], keep_prob: 1.0}
#accuracy.eval(feed_dict)