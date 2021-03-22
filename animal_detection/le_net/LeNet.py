from keras.models import Sequential
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras import backend as K

class LeNet : 
    @staticmethod
    def build(img_channel, img_width, img_height, num_classes, filter_num=[32, 32], filter_size=[5, 5], pooling_size = [(2,2), (2,2)], node_num = [256], activation = 'relu', weights_path=None) :
        if K.image_data_format() == 'channels_first' : 
            input_shape = (img_channel, img_height, img_width)
        else : 
            input_shape = (img_height, img_width, img_channel)

        model = Sequential()

        # Convolution Neural Network
        for i in range(len(filter_num)):
            model.add(Conv2D(filter_num[i], filter_size[i], input_shape = input_shape, padding = 'same'))
            model.add(Activation(activation))
            model.add(MaxPooling2D(pool_size=pooling_size[i], strides=pooling_size[i]))

        model.add(Flatten())

        # Full Connected network
        for i in range(len(node_num)) : 
            model.add(Dense(node_num[i]))
            model.add(Activation(activation))

        # Output layer
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        if weights_path is not None :
            print("[INFO] Loading model...")
            model.load_weights(weights_path)

        model.summary()
        return model