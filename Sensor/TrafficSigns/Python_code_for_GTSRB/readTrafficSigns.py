# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import cv2

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []# images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            img=plt.imread(prefix + row[0])
            #img=cv2.imread(prefix + row[0])
            imgs=cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
            image=np.array(imgs)
            #image.reshape((int('32'),int('32'),3))
            #print(image.shape)
            images.append(image) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
        print(c)
    return images, labels
#images,labels=readTrafficSigns('../GTSRB/Final_Training/Images')
images,labels=readTrafficSigns('../GTSRB/Final_Test/Images')
train={'features':images,'labels':labels}
f = open('test.p','wb')  
pickle.dump(train,f)  
f.close()  