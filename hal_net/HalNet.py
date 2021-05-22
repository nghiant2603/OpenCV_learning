from keras.models import Sequential
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from keras import backend as K

class HalNet : 
    @staticmethod
    def build_classification(img_channel, img_width, img_height, class_num, cov_filter_num=[32, 32], cov_filter_size=[5, 5], cov_activation = ['relu', 'relu'], cov_pooling_size = [(2,2), (2,2)], full_node_num = [256], full_activation = ['relu', 'softmax'], dropout=None, loss=None, opt=SGD(learning_rate=0.001), model_path=None) :
        '''
        Build the CNN classification model
            >>> Cov -> Activation -> Pooling -> Cov -> Activation -> Pooling ...-> Full connected layer <<<   \
            img_channel : image color channel   \
            img_width : image width \
            img_height : image height   \
            class_num : num of class in result  \
            cov_filter_num : list, num of filter in each covolution layer. Ex : [32, 32] --> 2 covolution layer,  32 filters/layer  \
            cov_filter_size  : filter size in each covolution layer \
            cov_activation : activation function in each covolution layer   \
            cov_pooling_size : pooling size in each covolution layer    \
            full_node_num : list, num of node in each full connected hidden layer. Ex : [128, 128] --> 2 hidden layer. 128nodes/layer   \
            full_activation : activation function in each full connected layer.     \
                Note : output layer activation may be included.     \
            dropout : dropout ratio. Default is None    \
            loss : loss function    \
            opt : optimize function \
            model_path : model result saving path 
        '''
        if K.image_data_format() == 'channels_first' : 
            input_shape = (img_channel, img_height, img_width)
        else : 
            input_shape = (img_height, img_width, img_channel)

        model = Sequential()

        # Convolution Neural Network
        if len(cov_filter_num) != len(cov_activation) : 
            print("[HALNET ERROR] Length of cov_activation, cov_filter_num, cov_polling_size and cov_filter_size must be equal ")
            return 0
        for i in range(len(cov_filter_num)):
            model.add(Conv2D(cov_filter_num[i], cov_filter_size[i], input_shape = input_shape, padding = 'same'))
            model.add(Activation(cov_activation[i]))
            model.add(MaxPooling2D(pool_size=cov_pooling_size[i], strides=cov_pooling_size[i]))

        model.add(Flatten())

        # Full Connected network
        for i in range(len(full_node_num)) : 
            model.add(Dense(full_node_num[i]))
            if dropout is not None : 
                model.add(Dropout(dropout[i]))
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

        if model_path is not None :
            print("[INFO] Loading model...")
            model.load_weights(model_path)

        model.summary()

        # Compile model
        if loss is not None : 
            model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        else : 
            model.compile(optimizer=opt, loss='binary_crossentropy' if (class_num == 2) else 'categorical_crossentropy', metrics=['accuracy'])
        return model