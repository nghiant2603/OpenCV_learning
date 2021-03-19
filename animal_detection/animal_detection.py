# import the necessary packages
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import LabelBinarizer
# from pyimagesearch.smallervggnet import SmallerVGGNet
# import matplotlib.pyplot as plt
# from imutils import paths
# import argparse
# import random
# import pickle
import os
from imutils import paths
import random
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataset = './ng_dataset'
data = []
labels = []
image_paths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(image_paths)
for i, img in enumerate(image_paths) :
    try : 
        frame = cv2.imread(img)
        #frame = img_to_array(frame)
        if frame.shape == (400, 400, 3) : 
            data.append(frame)
            labels.append(img.split(os.path.sep)[-2])
    except : 
        pass

data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size = 0.2, random_state = 42)

