from keras.models import Sequential
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.optimizers import SGD
from keras import backend as K

class HalNet : 
    @staticmethod
    def build_classification(img_channel, img_width, img_height, class_num, cov_filter_num=[32, 32], cov_filter_size=[5, 5], cov_pooling_size = [(2,2), (2,2)], cov_activation = ['relu'], full_node_num = [256], full_activation = ['relu'], loss=None, opt=SGD(learning_rate=0.001), weights_path=None) :
        if K.image_data_format() == 'channels_first' : 
            input_shape = (img_channel, img_height, img_width)
        else : 
            input_shape = (img_height, img_width, img_channel)

        model = Sequential()

        # Convolution Neural Network
        for i in range(len(cov_filter_num)):
            model.add(Conv2D(cov_filter_num[i], cov_filter_size[i], input_shape = input_shape, padding = 'same'))
            model.add(Activation(cov_activation[i]))
            model.add(MaxPooling2D(pool_size=cov_pooling_size[i], strides=cov_pooling_size[i]))

        model.add(Flatten())

        # Full Connected network
        for i in range(len(full_node_num)) : 
            model.add(Dense(full_node_num[i]))
            model.add(Activation(full_activation[i]))

        # Output layer
        model.add(Dense(class_num))
        if len(full_node_num) < len(full_activation) : 
            model.add(Activation(full_activation[-1]))
        else : 
            if class_num == 2 : 
                model.add(Activation('sigmoid'))
            else : 
                model.add(Activation('softmax'))

        if weights_path is not None :
            print("[INFO] Loading model...")
            model.load_weights(weights_path)

        model.summary()

        # Compile model
        if loss is not None : 
            model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        else : 
            model.compile(optimizer=opt, loss='binary_crossentropy' if (class_num == 2) else 'categorical_crossentropy', metrics=['accuracy'])
        return model