# Update from VM
from hal_net.HalNet import HalNet
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2
import time
import matplotlib.pyplot as plt

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# Load & pre-process mnist data
print ("[INFO] Load MNIST data...")
(train_data, train_label), (test_data, test_label) = mnist.load_data()

if K.image_data_format == 'channels_first' :
	train_data = train_data.reshape(train_data.shape[0], 1, 28, 28)
	test_data = test_data.reshape(test_data.shape[0], 1, 28, 28)
else:
	train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
	test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

train_data = train_data.astype("float32")/255.0
test_data = test_data.astype("float32")/255.0

train_label = np_utils.to_categorical(train_label, 10)
test_label = np_utils.to_categorical(test_label, 10)

# Compile model
opt = SGD(learning_rate=0.1)
model = HalNet.build_classification(1, 28, 28, 10, cov_filter_num=[20], cov_filter_size=[5], cov_pooling_size = [(2,2)], cov_activation = ['relu'], full_node_num = [100], full_activation = ['relu', 'softmax'], opt=opt, loss='categorical_crossentropy', weights_path=args['weights'] if args['load_model']>0 else None)

# Load/Train model
if args["load_model"] < 0 : 
	print ("[INFO] Training model...")
	s = time.time()
	his = model.fit(train_data, train_label, batch_size=500, epochs=10, verbose=1, validation_data=(test_data, test_label))
	t = time.time() - s
	print("[INFO] Training time : ", t)
	print(his.history.keys())
	plt.subplot(1, 2, 1)
	plt.plot(his.history['accuracy'])
	plt.plot(his.history['val_accuracy'])
	plt.legend(['accuracy', 'Val_accuracy'], loc='upper left')
	plt.title('Model accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('accuracy')
	plt.subplot(1, 2, 2)
	plt.plot(his.history['loss'])
	plt.plot(his.history['loss'])
	plt.plot(his.history['val_loss'])
	plt.legend(['loss', 'Val_loss'], loc='upper left')
	plt.title('Model loss')
	plt.xlabel('Epoch')
	plt.ylabel('loss')
	plt.show()

# Save model
if args["save_model"] > 0:
	print("[INFO] Saving model to file...")
	model.save_weights(args["weights"], overwrite=True)
