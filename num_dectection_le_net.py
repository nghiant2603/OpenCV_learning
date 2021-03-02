from le_net.LeNet import LeNet
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2
import time

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
opt = SGD(learning_rate=0.01)
model = LeNet.build(1, 28, 28, 10, filter_num=[50, 20], filter_size=[5, 5], pooling_size=[(2,2), (2, 2)], node_num=[500], weights_path=args['weights'] if args['load_model']>0 else None)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Load/Train model
if args["load_model"] < 0 : 
	print ("[INFO] Training model...")
	s = time.time()
	model.fit(train_data, train_label, batch_size=256, epochs=20, verbose=1)
	t = time.time() - s
	print("[INFO] Training time : ", t)
# Evaluate model
print ("[INFO] Evaluate model...")
(loss, accuracy) = model.evaluate(test_data, test_label, batch_size = 128, verbose = 1)
print ("[INFO] Accuracy {:.2f}%".format(accuracy*100))

# Save model
if args["save_model"] > 0:
	print("[INFO] Saving model to file...")
	model.save_weights(args["weights"], overwrite=True)