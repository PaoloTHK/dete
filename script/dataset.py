import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle

image_size = 256

#image = cv2.imread('right1.jpg')
#image1 = cv2.resize(image, (image_size, image_size))

train_path = 'train_image'
classes = ["left", "right"]
test = 588 #1470
num_classes = len(classes)
# def load_train(train_path, image_size, classes):
images = []
labels = []
ids = []
cls = []

def load_train(train_path, image_size, classes):
    
    images = []
    cls = []

    print('Reading training images')
    # data directory has a separate folder for each class, and that each folder is named after the class
    for folder in classes:   
        
        index = classes.index(folder)
        print('Loading {} files (Index: {})'.format(folder, index))
        path = os.path.join(train_path, folder, "*t*")
        
        print('path', path)
        files = glob.glob(path)
        
        for named_file in files:
            # to check weather right files are collecting
            print(named_file) 
            #resize as defined image_size for all data
            image = cv2.imread(named_file)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            #labels
	    label = np.zeros(int(test))
            label[index] = 1
            labels.append(label)
	    #classes
            images.append(image)
            cls.append(folder)
    
    images = np.array(images)
    cls = np.array(cls)

    return images, labels, cls

def read_train_sets(train_path, image_size, classes, validation_size=0):
  class DataSet(object):
    pass
  data_sets = DataSet()
  # seperate the validate file and training file for training as 2:8
  images, labels, cls = load_train(train_path, image_size, classes)
  # shuffle the data
  images, labels, cls = shuffle(images, labels, cls)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])
  
  # seperate as images as each training dataset
  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_cls = cls[validation_size:]

  # seperate as images as each validation set
  validation_images = images[:validation_size]
  validation_labes = labels[:validation_size]
  validation_cls = cls[:validation_size]

  return train_images, train_labels, train_cls
#load_train(train_path, image_size, classes)
