import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from keras import backend as K
import numpy as np
from sklearn.preprocessing import OneHotEncoder

(x_train_pre, y_train_pre), (x_test_pre, y_test_pre) = keras.datasets.mnist.load_data()

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train_pre.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test_pre.reshape(-1, 1)).toarray()

cnn_en = 0
if cnn_en : 
    ### 30s
    x_train = np.array(x_train_pre.reshape(-1, 28, 28, 1), dtype="float")/255.0
    x_test = np.array(x_test_pre.reshape(-1, 28, 28, 1), dtype="float")/255.0
    print("Length of Training data : {0}".format(len(x_train)))
    print("Length of Test data : {0}".format(len(x_test)))

    if K.image_data_format() == 'channels_first' : 
        input_shape = (1, 28, 28)
    else : 
        input_shape = (28, 28, 1)

    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = input_shape, activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=1250, epochs=10, verbose=1, validation_data=(x_test, y_test))
else : 
    ### 25s
    x_train = np.array(x_train_pre.reshape(-1, 784), dtype="float")/255.0
    x_test = np.array(x_test_pre.reshape(-1, 784), dtype="float")/255.0
    print("Length of Training data : {0}".format(len(x_train)))
    print("Length of Test data : {0}".format(len(x_test)))

    model = Sequential()
    model.add(Dense(4096, input_shape=(784, ), activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
