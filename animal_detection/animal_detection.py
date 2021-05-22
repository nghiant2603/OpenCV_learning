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

dataset = 'C:\\Users\\HP\\Documents\\200_DATABASE\\database\\animal_detection'
data = []
labels = []
image_paths = sorted(list(paths.list_images(dataset)))
print ("[INFO] : Total of Image : {0}".format(len(image_paths)))
random.seed(42)
random.shuffle(image_paths)
for i, img in enumerate(image_paths) :
    try : 
        frame = cv2.imread(img)
        frame = cv2.resize(frame, (50, 50), interpolation = cv2.INTER_AREA)
        if frame.shape == (50, 50, 3) : 
            data.append(frame)
            labels.append(img.split(os.path.sep)[-2])
    except : 
        pass

data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

(train_data, test_data, train_label, test_label) = train_test_split(data, labels, test_size = 0.2, random_state = 42)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=str, default=None,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=str, default=None,
	help="(optional) whether or not pre-trained model should be loaded")
args = vars(ap.parse_args())

# Compile model
opt = SGD(learning_rate=0.05)
model = HalNet.build_classification(3, 50, 50, 3, cov_filter_num=[32, 32], cov_filter_size=[5, 5], cov_pooling_size = [(2,2), (2,2)], cov_activation = ['relu', 'relu'], full_node_num = [16], full_activation = ['relu', 'softmax'], dropout=None, loss="categorical_crossentropy", opt=SGD(learning_rate=0.001), model_path=args['load_model'])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Load/Train model
if args["load_model"] is None : 
	print ("[INFO] Training model...")
	s = time.time()
	his = model.fit(train_data, train_label, batch_size=16, epochs=30, verbose=1, validation_data=(test_data, test_label))
	t = time.time() - s
	print("[INFO] Training time : {0}".format(t))
	plt.plot(his.history['accuracy'])
	plt.plot(his.history['val_accuracy'])
	plt.legend(['accuracy', 'val_accuracy'])
	plt.show()

#demofiles = glob.glob("./example/" + "*.jpg")
demofiles = list(paths.list_images('./example/'))
for i in demofiles : 
	demoframe = cv2.imread(i)
	demoframe = cv2.resize(demoframe, (50, 50), interpolation=cv2.INTER_AREA)
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
