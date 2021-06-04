from keras.models import Sequential
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
from keras import backend as K
from tensorflow.python.keras.backend import batch_normalization

class HalNet : 
    @staticmethod
    def build_classification(img_channel, img_width, img_height, class_num, cov_filter_num=[32, 32], cov_filter_size=[5, 5], cov_activation = ['relu', 'relu'], cov_pooling_size = [(2,2), (2,2)], cov_batchnor = False, full_node_num = [256], full_activation = ['relu', 'softmax'], dropout=0, regular=0, f_batchnor = False, loss=None, opt=SGD(learning_rate=0.001), model_path=None) :
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
            cov_batchnor : batch normalization enable for Cov layer         \
            full_node_num : list, num of node in each full connected hidden layer. Ex : [128, 128] --> 2 hidden layer. 128nodes/layer   \
            full_activation : activation function in each full connected layer.     \
                Note : output layer activation may be included.     \
            dropout : dropout ratio for each full connected layer. Default is 0    \
            regular : L2 regular value for each full connected layer. Default is 0    \
            f_batchnor : batch normalization enable for full connected layer         \
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
        for i in range(len(cov_filter_num)):
            model.add(Conv2D(cov_filter_num[i], cov_filter_size[i], input_shape = input_shape, padding = 'same', kernel_initializer='random_normal', bias_initializer='zeros'))
            if cov_batchnor : 
                model.add(BatchNormalization())
            model.add(Activation(cov_activation[i]))
            model.add(MaxPooling2D(pool_size=cov_pooling_size[i], strides=cov_pooling_size[i]))

        model.add(Flatten())

        # Full Connected network
        for i in range(len(full_node_num)) : 
            if regular != 0 : 
                model.add(Dense(full_node_num[i], kernel_regularizer=l2(regular), kernel_initializer='random_normal', bias_initializer='zeros'))
            else : 
                model.add(Dense(full_node_num[i], kernel_initializer='random_normal', bias_initializer='zeros'))
            if dropout != 0 : 
                model.add(Dropout(dropout))
            if f_batchnor : 
                model.add(BatchNormalization())
            model.add(Activation(full_activation[i]))

        # Output layer
        model.add(Dense(class_num, kernel_initializer='random_normal', bias_initializer='zeros'))
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