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
from tensorflow.random import set_seed
set_seed(20)
import matplotlib.pyplot as plt
import glob
from numpy import argmax
import pickle
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import keras

cifar3 = 1

if cifar3 : 
	cifar_labels = ['bird', 'cat', 'dog']
else : 
	cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

### Load CIFAR data 
#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

if cifar3 :
	(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
	x_train_bird = x_train[np.where(y_train == 2)[0]] 
	x_train_cat = x_train[np.where(y_train == 3)[0]]  
	x_train_dog = x_train[np.where(y_train == 5)[0]] 
	y_train_bird = y_train[np.where(y_train == 2)[0]] 
	y_train_cat = y_train[np.where(y_train == 3)[0]]  
	y_train_dog = y_train[np.where(y_train == 5)[0]] 
	x_train_pre = np.concatenate((x_train_bird, x_train_cat, x_train_dog))
	y_train_pre = np.concatenate((y_train_bird, y_train_cat, y_train_dog))

	x_test_bird = x_test[np.where(y_test == 2)[0]] 
	x_test_cat = x_test[np.where(y_test == 3)[0]]  
	x_test_dog = x_test[np.where(y_test == 5)[0]] 
	y_test_bird = y_test[np.where(y_test == 2)[0]] 
	y_test_cat = y_test[np.where(y_test == 3)[0]]  
	y_test_dog = y_test[np.where(y_test == 5)[0]] 
	x_test_pre = np.concatenate((x_test_bird, x_test_cat, x_test_dog))
	y_test_pre = np.concatenate((y_test_bird, y_test_cat, y_test_dog))

	x_train_pro, y_train_pro = shuffle(x_train_pre, y_train_pre)
	x_test_pro, y_test_pro = shuffle(x_test_pre, y_test_pre)
else : 
	(x_train_pro, y_train_pro), (x_test_pro, y_test_pro) = keras.datasets.cifar10.load_data()

x_train = np.array(x_train_pro, dtype="float")/255.0
x_test = np.array(x_test_pro, dtype="float")/255.0

print("Length of Training data : {0}".format(x_train.shape[0]))
print("Length of Testing data : {0}".format(x_test.shape[0]))
enc = OneHotEncoder()
y_train_ca = enc.fit_transform(y_train_pro).toarray()
y_test_ca = enc.transform(y_test_pro).toarray()

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
opt = SGD(learning_rate=0.1)
if cifar3 : 
	model = HalNet.build_classification(3, 32, 32, 3, cov_filter_num=[16, 8, 8], cov_filter_size=[5, 5, 5], cov_pooling_size = [(2,2), (2,2), (2,2)], cov_activation = ['relu', 'relu', 'relu'], cov_batchnor = False, full_node_num = [16], full_activation = ['relu', 'softmax'], dropout=0.5, regular=0, f_batchnor = False, loss="categorical_crossentropy", opt=opt, model_path=args['load_model'])
else : 
	model = HalNet.build_classification(3, 32, 32, 10, cov_filter_num=[32, 32], cov_filter_size=[5, 5], cov_pooling_size = [(2,2), (2,2)], cov_activation = ['relu', 'relu'], cov_batchnor = False, full_node_num = [16], full_activation = ['relu', 'softmax'], dropout=0, regular=0, f_batchnor = False, loss="categorical_crossentropy", opt=opt, model_path=args['load_model'])

callback = keras.callbacks.EarlyStopping( monitor="val_loss", min_delta=0.01, patience=10, verbose=0, mode="auto", baseline=None, restore_best_weights=True)

# Load/Train model
if args["load_model"] is None : 
	print ("[INFO] Training model...")
	s = time.time()
	if cifar3 : 
		#his = model.fit(x_train, y_train_ca, batch_size=100, epochs=100, verbose=1, validation_data=(x_test, y_test_ca), callbacks = [callback])
		his = model.fit(x_train, y_train_ca, batch_size=100, epochs=100, verbose=1, validation_data=(x_test, y_test_ca))
	else : 
		his = model.fit(x_train, y_train_ca, batch_size=100, epochs=100, verbose=1, validation_data=(x_test, y_test_ca), callbacks = [callback])
	t = time.time() - s
	print("[INFO] Training time : {0}".format(t))

	plt.subplot(1, 2, 1)
	plt.plot(his.history['accuracy'])
	plt.plot(his.history['val_accuracy'])
	plt.legend(['accuracy', 'val_accuracy'])
	plt.subplot(1, 2, 2)
	plt.plot(his.history['loss'])
	plt.plot(his.history['val_loss'])
	plt.legend(['loss', 'val_loss'])
	if args['debug'] is not None : 
		if not os.path.exists("debug") : 
			os.mkdir("debug")
		plt.savefig("debug" + os.sep + args['debug'], dpi=300)
	else : 
		plt.show()

#demofiles = glob.glob("./example/" + "*.jpg")
demofiles = list(paths.list_images('./example/'))
for i in demofiles : 
	demoframe = cv2.imread(i)
	demoframe = cv2.resize(demoframe, (32, 32), interpolation=cv2.INTER_AREA)
	demoframe = np.array(demoframe, dtype="float")/255.0
	categorical_result = model.predict(demoframe.reshape(1, 32, 32, 3))
	label_encorded_result = argmax(categorical_result)
	print("[INFO] Predict  {0} :: FILE {1}".format(cifar_labels[label_encorded_result], i))


# Evaluate model
#print ("[INFO] Evaluate model...")
#(loss, accuracy) = model.evaluate(test_data, test_label, batch_size = 16, verbose = 1)
#print ("[INFO] Accuracy {:.2f}%".format(accuracy*100))

# Save model
if args["save_model"] is not None:
	print("[INFO] Saving model to *.h5 file...")
	model.save_weights(args["save_model"], overwrite=True)
