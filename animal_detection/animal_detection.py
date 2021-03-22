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
from le_net.LeNet import LeNet
import argparse
from keras.optimizers import SGD
import time

dataset = 'C:\\Users\\HP\\Documents\\200_DATABASE\\database\\animals_detection'
data = []
labels = []
image_paths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(image_paths)
for i, img in enumerate(image_paths) :
    try : 
        frame = cv2.imread(img)
        #frame = img_to_array(frame)
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
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# Compile model
opt = SGD(learning_rate=0.05)
model = LeNet.build(3, 50, 50, 3, filter_num=[32, 32, 32], filter_size=[5, 5, 5], pooling_size=[(2,2), (2, 2), (2,2)], node_num=[16], weights_path=args['weights'] if args['load_model']>0 else None)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Load/Train model
if args["load_model"] < 0 : 
	print ("[INFO] Training model...")
	s = time.time()
	model.fit(train_data, train_label, batch_size=16, epochs=100, verbose=1)
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
