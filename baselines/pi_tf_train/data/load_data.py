import sys, os, time
import cv2
import glob
import math, random
# Load data

def light_load_img(image_types,base_image_path):
    full_set = []
    for im_type in image_types:
        for ex in glob.glob(os.path.join(base_image_path, im_type, "*")):
            im = cv2.imread(ex)
            if not im is None:
                im = cv2.resize(im, (32, 32))
                one_hot_array = [0] * len(image_types)
                one_hot_array[image_types.index(im_type)] = 1
                assert(im.shape == (32, 32, 3))
                full_set.append((im, one_hot_array, ex))
    random.shuffle(full_set)
    return full_set

def light_split(full_set,train_test_split_ratio,batch_size):


    split_index = int(math.floor(len(full_set) * train_test_split_ratio))
    train_set = full_set[:split_index]
    test_set = full_set[split_index:]

    train_set_offset = len(train_set) % batch_size
    test_set_offset = len(test_set) % batch_size
    train_set = train_set[: len(train_set) - train_set_offset]
    test_set = test_set[: len(test_set) - test_set_offset]

    train_x, train_y, train_z = zip(*train_set)
    test_x, test_y, test_z = zip(*test_set)
    return train_x, train_y,train_z, test_x, test_y,test_z


#from sklearn.utils import shuffle

#X_train, y_train = shuffle(X_train, y_train)