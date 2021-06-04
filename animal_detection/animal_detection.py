import os
import sys
sys.path.append(os.path.abspath("C:\\Users\\HP\\Documents\\100_TECH\\100_CODE\\100_TRAIN\\300_ComputerVision\\hal_net"))

from imutils import paths
import random
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from HalNet import *
import argparse
from keras.optimizers import SGD
import time
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
import matplotlib.pyplot as plt
import glob
from numpy import argmax

import numpy as np

### Load CIFAR data 
#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#train_dataset = 'C:\\Users\\HP\\Documents\\200_DATABASE\\database\\animal_detection_rotation\\train_data'
#test_dataset = 'C:\\Users\\HP\\Documents\\200_DATABASE\\database\\animal_detection_rotation\\test_data'
train_dataset = 'C:\\Users\\HP\\Documents\\200_DATABASE\\database\\animal_detection\\train_data'
test_dataset = 'C:\\Users\\HP\\Documents\\200_DATABASE\\database\\animal_detection\\test_data'
train_images = sorted(list(paths.list_images(train_dataset)))
test_images = sorted(list(paths.list_images(test_dataset)))
print ("[INFO] : Total of Train Image : {0}".format(len(train_images)))
print ("[INFO] : Total of Test Image : {0}".format(len(test_images)))
random.seed(42)
random.shuffle(train_images)
random.shuffle(test_images)
train_datas = []
train_labels = []
test_datas = []
test_labels = []
for i, img in enumerate(train_images) :
    try : 
        frame = cv2.imread(img)
        frame = cv2.resize(frame, (50, 50), interpolation = cv2.INTER_AREA)
        if frame.shape == (50, 50, 3) : 
            train_datas.append(frame)
            train_labels.append(img.split(os.path.sep)[-2])
    except : 
        pass

for i, img in enumerate(test_images) :
    try : 
        frame = cv2.imread(img)
        frame = cv2.resize(frame, (50, 50), interpolation = cv2.INTER_AREA)
        if frame.shape == (50, 50, 3) : 
            test_datas.append(frame)
            test_labels.append(img.split(os.path.sep)[-2])
    except : 
        pass

train_datas = np.array(train_datas, dtype="float")/255.0
train_labels = np.array(train_labels)
test_datas = np.array(test_datas, dtype="float")/255.0
test_labels = np.array(test_labels)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
train_labels = to_categorical(train_labels)
test_labels = label_encoder.transform(test_labels)
test_labels = to_categorical(test_labels)

#(train_data, test_data, train_label, test_label) = train_test_split(data, labels, test_size = 0.2, random_state = 42)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=str, default=None,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=str, default=None,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-d", "--debug", type=str, default=None,
	help="(optional) where to save debug data")
args = vars(ap.parse_args())

# Compile model
opt = SGD(learning_rate=0.05)
model = HalNet.build_classification(3, 50, 50, 3, cov_filter_num=[32, 32], cov_filter_size=[5, 5], cov_pooling_size = [(2,2), (2,2)], cov_activation = ['relu', 'relu'], full_node_num = [16], full_activation = ['relu', 'softmax'], dropout=None, regular=[0.3], loss="categorical_crossentropy", opt=opt, model_path=args['load_model'])
model.summary()

# Load/Train model
if args["load_model"] is None : 
	print ("[INFO] Training model...")
	s = time.time()
	his = model.fit(train_datas, train_labels, batch_size=16, epochs=100, verbose=1, validation_data=(test_datas, test_labels))
	t = time.time() - s
	print("[INFO] Training time : {0}".format(t))

	plt.subplot(2, 1, 1)
	plt.plot(his.history['accuracy'])
	plt.plot(his.history['val_accuracy'])
	plt.legend(['accuracy', 'val_accuracy'])
	plt.subplot(2, 1, 2)
	plt.plot(his.history['loss'])
	plt.plot(his.history['val_loss'])
	plt.legend(['loss', 'val_loss'])
	if args['debug'] is not None : 
		if not os.path.exists(args['debug']) : 
			os.mkdir(args['debug'])
		plt.savefig(args['debug'] + os.sep + 'model_train_result.jpg', dpi=300)
	else : 
		plt.show()

#demofiles = glob.glob("./example/" + "*.jpg")
demofiles = list(paths.list_images('./example/'))
for i in demofiles : 
	demoframe = cv2.imread(i)
	demoframe = cv2.resize(demoframe, (50, 50), interpolation=cv2.INTER_AREA)
	demoframe = np.array(demoframe, dtype="float")/255.0
	categorical_result = model.predict(demoframe.reshape(1, 50, 50, 3))
	label_encorded_result = argmax(categorical_result)
	print("[INFO] Predict  {0} :: FILE {1}".format(label_encoder.inverse_transform(label_encorded_result.reshape(1,)), i))


# Evaluate model
#print ("[INFO] Evaluate model...")
#(loss, accuracy) = model.evaluate(test_data, test_label, batch_size = 16, verbose = 1)
#print ("[INFO] Accuracy {:.2f}%".format(accuracy*100))

# Save model
if args["save_model"] is not None:
	print("[INFO] Saving model to *.h5 file...")
	model.save_weights(args["save_model"], overwrite=True)
