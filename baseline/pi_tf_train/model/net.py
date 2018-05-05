
# coding: utf-8
import tensorflow as tf
from matplotlib import pyplot as plt
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def layblock(x,k,inSize,outSize,isPool=True):
    W_conv = weight_variable([k[0], k[1], inSize, outSize])
    b_conv = bias_variable([outSize])
    h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
    if isPool:
        h_pool = max_pool_2x2(h_conv)
    else:
        h_pool = h_conv
    return h_pool
"""
def weight_variable(scope,shape):
    with tf.variable_scope(scope):
        W = tf.get_variable('W',shape,initializer=tf.contrib.layers.xavier_initializer())
        return W


def bias_variable(scope,shape):
    with tf.variable_scope(scope):
        b = tf.get_variable('b', shape, initializer=tf.constant_initializer(0.1))
        return b
"""
def batch_norm_conv_layer(scope,input, weight_shape, phase):
    with tf.variable_scope(scope):
        W_conv = weight_variable(scope,weight_shape)
        b_conv = bias_variable(scope,[weight_shape[-1]])
        h_conv = conv2d(input, W_conv) + b_conv
        is_training = True if phase is not None else False
        h2 = tf.contrib.layers.batch_norm(h_conv,
                                          center=True, scale=True,
                                          is_training=is_training)
    return h2


# TODO: consolidate this somehow into batch_norm_conv_layer
def batch_norm_pool_conv_layer(scope,input, weight_shape, phase):
    with tf.variable_scope(scope):
        W_conv = weight_variable(scope,weight_shape)
        b_conv = bias_variable(scope,[weight_shape[-1]])
        h_conv = conv2d(input, W_conv) + b_conv
        max_pool = max_pool_2x2(tf.nn.relu(h_conv))
        is_training = True if phase is not None else False
        h2 = tf.contrib.layers.batch_norm(max_pool,
                                          center=True, scale=True,
                                          is_training=is_training)
    return h2


def batch_norm_fc_layer(scope,input, weight_shape, phase):
    with tf.variable_scope(scope):
        W = weight_variable(scope,weight_shape)
        b = bias_variable(scope,[weight_shape[-1]])
        h = tf.nn.relu(tf.matmul(input, W) + b)
        is_training = True if phase is not None else False
        h2 = tf.contrib.layers.batch_norm(h,
                                          center=True, scale=True,
                                          is_training=is_training)
    return h2


def flat(h_pool4):
    n1, n2, n3, n4 = h_pool4.get_shape().as_list()

    W_fc1 = weight_variable([n2*n3*n4, 3])
    b_fc1 = bias_variable([3])

    # We flatten our pool layer into a fully connected layer

    h_pool4_flat = tf.reshape(h_pool4, [-1, n2*n3*n4])

    y = tf.matmul(h_pool4_flat, W_fc1) + b_fc1
    return y


def deepnn(x):
# 最后一个维度的特征数量
#  RGB 是三个通道, RGBA是四个通道.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一个层 1通道输入 32通道输出
    block1=layblock(x_image,[5,5],1,32)
    block2=layblock(block1,[5,5],32,64)



# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
# is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(block2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout - controls the complexity of the model, prevents co-adaptation of
# features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def ann(x):
    x_shaped = tf.reshape(x, [-1, 240 * 320 * 3])

    W1 = weight_variable('layer1',[240 * 320 * 3, 32])
    b1 = bias_variable('layer1',[32])
    h1 = tf.sigmoid(tf.matmul(x_shaped, W1) + b1)

    W2 = weight_variable('layer2',[32, 3])
    b2 = bias_variable('layer2',[3])
    pred=tf.matmul(h1, W2) + b2
    return pred

def light_net(x_image):
    block1=layblock(x_image,[3,3],3,16,False)
    block2=layblock(block1,[3,3],16,16,False)
    block3=layblock(block2,[3,3],16,16,True)
    y=flat(block3)
    return y

def optimizer(pred,y_):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-5,name='train_step').minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

"""

def show(img):
    plt.figure()
    plt.imshow(img,cmap=plt.cm.gray)
    plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    #plt.savefig("gray.png")
"""
def show(img):
    plt.figure()
    plt.imshow(img)
    plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    #plt.savefig("gray.png")