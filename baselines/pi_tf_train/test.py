# coding: utf-8
import tensorflow as tf
import pickle
import numpy as np

import time
from model.net import deepnn,show

x = tf.placeholder(tf.float32, [None, 784])

y_conv, keep_prob = deepnn(x)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

pkl_file = open('./data/test_datas.pkl', 'rb')
test_images,test_labels = pickle.load(pkl_file,encoding='iso-8859-1')

test_images=test_images[400]
  
test_img=np.reshape(test_images, [-1, 784])

saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./pretrain/')
    saver.restore(sess,ckpt.model_checkpoint_path)
    start=time.time()
    result = sess.run(y_conv, feed_dict={x: test_img,keep_prob: 1.0})
    #print(result)
    print(sess.run(tf.argmax(result,1)))

end=time.time()

s=end-start
print(s)
img=np.reshape(test_images,[28,28])
show(img)



"""
## load the model for this network 
## the model is a kind of weighs and biases excactly which are optimized
## during training procedure on the faster computer such as GPU embedded
## cloud instance

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
       
saver = tf.train.Saver()
saver.restore(sess, './model/model' )         


## print the prediction results of test datas
## 0.97 should be printed
## 

feed_dict = {x: test_images[:100], y_: test_labels[:100], keep_prob: 1.0}
print(accuracy.eval(feed_dict))
"""






