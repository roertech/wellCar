import cv2
import sys,os, time,random,glob,math
import h5py
import numpy as np
import tensorflow as tf

input_img_x, input_img_y=32,32
image_types = ["red", "green", "yellow"]
checkpoint_name = "model.ckpt"
def save_h5():
    image_path="5_tensorflow_traffic_light_images"
    image_type=["red","green","yellow"]
    full_set=[]
    label_set=[]
    for im_type in image_type:
        for ex in glob.glob(os.path.join(image_path,im_type,"*")):
            im=cv2.imread(ex)
            if not im is None:
                im=cv2.resize(im,(32,32))
                one_hot=[0]*len(image_type)
                one_hot[image_type.index(im_type)]=1
                assert(im.shape==(32,32,3))
                #full_set.append((im,one_hot,ex))
                full_set.append(im)
                label_set.append(one_hot)
    #random.shuffle(full_set)
    full_set=np.array(full_set)
    label_set=np.array(label_set)
    print(full_set.shape)
    print(label_set.shape)
    file = h5py.File('Traffic_light.h5','w')
    file.create_dataset('full_x',data=full_set)
    file.create_dataset('full_y',data=label_set)
    file.close()

def load_h5():
    file=h5py.File('Traffic_light.h5','r')
    full_x = file['full_x'][:]
    full_y = file['full_y'][:]
    file.close()
    return full_x,full_y

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



def model():
    x = tf.placeholder(tf.float32, shape=[None, input_img_x, input_img_y, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, len(image_types)])


    x_image = x

    # Our first three convolutional layers, of 16 3x3 filters
    W_conv1 = weight_variable([3, 3, 3, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 1) + b_conv1)

    W_conv2 = weight_variable([3, 3, 16, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 1) + b_conv2)

    W_conv3 = weight_variable([3, 3, 16, 16])
    b_conv3 = bias_variable([16])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    # Our pooling layer

    h_pool4 = max_pool_2x2(h_conv3)

    n1, n2, n3, n4 = h_pool4.get_shape().as_list()

    W_fc1 = weight_variable([n2*n3*n4, 3])
    b_fc1 = bias_variable([3])
    # We flatten our pool layer into a fully connected layer
    h_pool4_flat = tf.reshape(h_pool4, [-1, n2*n3*n4])
    y = tf.matmul(h_pool4_flat, W_fc1) + b_fc1
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return loss,train_step,x,y,y_

def batch_dataset(full_sets,label_sets):
    train_test_split_ratio=0.7
    split_index = int(math.floor(len(full_sets) * train_test_split_ratio))
    train_set = full_sets[:split_index]
    test_set = full_sets[split_index:]

    train_set_offset = len(train_set) % batch_size
    test_set_offset = len(test_set) % batch_size
    train_x = train_set[: len(train_set) - train_set_offset]
    test_x = test_set[: len(test_set) - test_set_offset]

    
    split_index = int(math.floor(len(label_sets) * train_test_split_ratio))
    train_label = label_sets[:split_index]
    test_label = label_sets[split_index:]

    train_set_offset = len(train_label) % batch_size
    test_set_offset = len(test_label) % batch_size
    train_y = train_label[: len(train_label) - train_set_offset]
    test_y = test_label[: len(test_label) - test_set_offset]
  
    return train_x,test_x,train_y,test_y

if __name__ == '__main__':
    #save_h5()
    
    batch_size = 32
    full_set,label=load_h5()
   

   
    #print(pp[:,1])
    train_x,test_x,train_y,test_y=batch_dataset(full_set,label)
   
    print('train_x',train_x.shape)
    print('test_x',test_x.shape)
    print('train_y',train_y.shape)
    print('test_y',test_y.shape)
   

    
    sess = tf.InteractiveSession()

    loss,train_step,x,y,y_=model()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
             batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
             accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
             total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    time_start = time.time()
    max_epochs = 250
    least_loss = 99999999
    v_loss = 9999999
    
    train_loss = []
    val_loss = []
    
    for i in range(0, max_epochs):

    # Iterate over our training set
        for tt in range(0, (len(train_x) / batch_size)):
            start_batch = batch_size * tt
            end_batch = batch_size * (tt + 1)
            train_step.run(feed_dict={x: train_x[start_batch:end_batch], y_: train_y[start_batch:end_batch]})
            ex_seen = "Current epoch, examples seen: {:20} / {} \r".format(tt * batch_size, len(train_x))
            sys.stdout.write(ex_seen.format(tt * batch_size))
            sys.stdout.flush()

        ex_seen = "Current epoch, examples seen: {:20} / {} \r".format((tt + 1) * batch_size, len(train_x))
        sys.stdout.write(ex_seen.format(tt * batch_size))
        sys.stdout.flush()

        t_loss = loss.eval(feed_dict={x: train_x, y_: train_y})
        v_loss = loss.eval(feed_dict={x: test_x, y_: test_y})
        validation_accuracy = evaluate(test_x, test_y)
        #print("EPOCH {} ... Validation Accuracy = {:.3f}".format(i+1,validation_accuracy))
        #print()
        train_loss.append(t_loss)
        val_loss.append(v_loss)

        sys.stdout.write("Epoch {:5}: loss: {:15.10f}, val. loss: {:15.10f},Validation Accuracy: {:.3f}".format(i + 1, t_loss, v_loss,validation_accuracy))

        if v_loss < least_loss:
            sys.stdout.write(", saving new best model to {}".format(checkpoint_name))
            least_loss = v_loss
            filename = saver.save(sess, checkpoint_name)

        sys.stdout.write("\n")
    
"""
    import matplotlib.image as mpimg

    new_sign = ['sign1.jpg', 'sign2.jpg', 'sign3.jpg', 'sign4.jpg', 'sign5.jpg']
    
    test_img = []

    for img in new_sign:
        image = mpimg.imread('new_signs/' + img)
        image = cv2.resize(image,(32,32))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        test_img.append(image)    
    
    plt.figure(figsize=(12,1))
    for i in range(5):
       plt.subplot(1,5,i+1)
       plt.imshow(test_img[i])

    softmax = tf.nn.softmax(logits)
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph('model.ckpt.meta')
        loader.restore(sess, tf.train.latest_checkpoint('./'))
    
        k = 3
        
        my_prediction = sess.run(softmax, feed_dict={x:test_img})
        top_k = tf.nn.top_k(softmax, k)

        #output = sess.run(softmax, feed_dict={x: test_img})
        #top3_pred = sess.run(tf.nn.top_k((output), k=3))
        values, indices = sess.run(top_k, feed_dict={x:test_img}) 
   
"""

