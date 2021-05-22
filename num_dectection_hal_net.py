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
opt = SGD(learning_rate=0.05)
model = {}
model["Cov_20_20_full_500"] 			= HalNet.build_classification(1, 28, 28, 10, cov_filter_num=[20, 20], cov_filter_size=[5, 5], cov_pooling_size = [(2,2), (2,2)], cov_activation = ['relu', 		'relu'], 	full_node_num = [500], full_activation = ['relu', 		'softmax'], opt=opt, loss='categorical_crossentropy', weights_path=args['weights'] if args['load_model']>0 else None)
model["Cov_20_20_full_500_dropout_0p1"] = HalNet.build_classification(1, 28, 28, 10, cov_filter_num=[20, 20], cov_filter_size=[5, 5], cov_pooling_size = [(2,2), (2,2)], cov_activation = ['relu', 		'relu'], 	full_node_num = [500], full_activation = ['relu', 		'softmax'], dropout=[0.1], opt=opt, loss='categorical_crossentropy', weights_path=args['weights'] if args['load_model']>0 else None)
model["Cov_20_20_full_500_dropout_0p3"] = HalNet.build_classification(1, 28, 28, 10, cov_filter_num=[20, 20], cov_filter_size=[5, 5], cov_pooling_size = [(2,2), (2,2)], cov_activation = ['relu', 		'relu'], 	full_node_num = [500], full_activation = ['relu', 		'softmax'], dropout=[0.3], opt=opt, loss='categorical_crossentropy', weights_path=args['weights'] if args['load_model']>0 else None)
model["Cov_20_20_full_500_dropout_0p5"] = HalNet.build_classification(1, 28, 28, 10, cov_filter_num=[20, 20], cov_filter_size=[5, 5], cov_pooling_size = [(2,2), (2,2)], cov_activation = ['relu', 		'relu'], 	full_node_num = [500], full_activation = ['relu', 		'softmax'], dropout=[0.5], opt=opt, loss='categorical_crossentropy', weights_path=args['weights'] if args['load_model']>0 else None)
model["Cov_20_20_full_500_dropout_0p7"] = HalNet.build_classification(1, 28, 28, 10, cov_filter_num=[20, 20], cov_filter_size=[5, 5], cov_pooling_size = [(2,2), (2,2)], cov_activation = ['relu', 		'relu'], 	full_node_num = [500], full_activation = ['relu', 		'softmax'], dropout=[0.7], opt=opt, loss='categorical_crossentropy', weights_path=args['weights'] if args['load_model']>0 else None)

# Load/Train model
if args["load_model"] < 0 : 
	print ("[INFO] Training model...")
	his = {}	
	i = 0
	for k in model.keys() :
		print ("[INFO] Training model {0}...".format(k))
		t = time.time()
		his[k] = model[k].fit(train_data[:5000], train_label[:5000], batch_size=500, epochs=100, verbose=2, validation_data=(test_data, test_label))
		plt.subplot(len(model.keys()), 2, 2*i + 1)
		plt.ylim(0,1)
		plt.plot(his[k].history['accuracy'])
		plt.plot(his[k].history['val_accuracy'])
		plt.legend([k + '_accuracy', k + '_Val_accuracy'], loc='lower right')
		if i == (len(model.keys()) - 1) : 
			plt.xlabel('Epoch')
			plt.ylabel('accuracy')
		plt.subplot(len(model.keys()), 2, 2*i + 2)
		plt.plot(his[k].history['loss'])
		plt.plot(his[k].history['val_loss'])
		plt.legend([k + '_loss', k + '_Val_loss'], loc='upper right')
		if i == (len(model.keys()) - 1) : 
			plt.xlabel('Epoch')
			plt.ylabel('loss')
		i += 1
		running_time = time.time() - t
		print("Running Time : {0}s".format(running_time))
	plt.show()

# Save model
if args["save_model"] > 0:
	print("[INFO] Saving model to file...")
	model.save_weights(args["weights"], overwrite=True)
