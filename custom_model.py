import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, MaxPooling1D, AveragePooling1D 
from tensorflow.keras.layers import Conv2D, Conv1D 
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D, SpatialDropout1D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    # start the model
    
    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    # flatten      = Flatten()(block1)
    # dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    # softmax      = Activation('softmax')(dense)
    
    # return Model(inputs=input_main, outputs=softmax)
    return block1

def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    

    #temperal conv
    b1_block1       = Conv2D(64, (1, 512), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input_main)
    b1_block1       = BatchNormalization()(block1)
    #spatial conv
    b1_block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    b1_block1       = BatchNormalization()(block1)
    b1_block1       = Activation('elu')(block1)
    b1_block1       = AveragePooling2D((1, 4))(block1)
    b1_block1       = Dropout(dropoutRate)(block1)
    
    #temporal + spatial conv
    b1_block2       = SeparableConv2D(128, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    b1_block2       = BatchNormalization()(block2)
    b1_block2       = Activation('elu')(block2)
    b1_block2       = AveragePooling2D((1, 8))(block2)
    b1_block2       = Dropout(dropoutRate)(block2)
    # flatten      = Flatten(name = 'flatten')(block2)
    
    # dense        = Dense(nb_classes, name = 'dense', 
    #                      kernel_constraint = max_norm(norm_rate))(flatten)
    # softmax      = Activation('softmax', name = 'softmax')(dense)
    
    # return Model(inputs=input1, outputs=softmax)
    return block2

if __name__ == "__main__":
    input_main   = Input((Chans, Samples, 1))
    
    #BRANCH - 1 ------------------------------------------------------------------
    b1_block1       = Conv2D(64, (1, 512), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input_main)
    b1_block1       = BatchNormalization()(block1)
    #spatial conv
    b1_block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    b1_block1       = BatchNormalization()(block1)
    b1_block1       = Activation('elu')(block1)
    b1_block1       = AveragePooling2D((1, 4))(block1)
    b1_block1       = Dropout(dropoutRate)(block1)
    
    #temporal + spatial conv
    b1_block2       = SeparableConv2D(128, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    b1_block2       = BatchNormalization()(block2)
    b1_block2       = Activation('elu')(block2)
    b1_block2       = AveragePooling2D((1, 8))(block2)
    b1_block2       = Dropout(dropoutRate)(block2)
    
    #BRANCH - 2 ------------------------------------------------------------------
    b1_block1       = Conv2D(64, (1, 256), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input_main)
    b1_block1       = BatchNormalization()(block1)
    #spatial conv
    b1_block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    b1_block1       = BatchNormalization()(block1)
    b1_block1       = Activation('elu')(block1)
    b1_block1       = AveragePooling2D((1, 4))(block1)
    b1_block1       = Dropout(dropoutRate)(block1)
    
    #temporal + spatial conv
    b1_block2       = SeparableConv2D(64, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    b1_block2       = BatchNormalization()(block2)
    b1_block2       = Activation('elu')(block2)
    b1_block2       = AveragePooling2D((1, 8))(block2)
    b1_block2       = Dropout(dropoutRate)(block2)
    
    #BRANCH - 3 ------------------------------------------------------------------
    b1_block1       = Conv2D(64, (1, 128), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input_main)
    b1_block1       = BatchNormalization()(block1)
    #spatial conv
    b1_block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    b1_block1       = BatchNormalization()(block1)
    b1_block1       = Activation('elu')(block1)
    b1_block1       = AveragePooling2D((1, 4))(block1)
    b1_block1       = Dropout(dropoutRate)(block1)
    
    #temporal + spatial conv
    b1_block2       = SeparableConv2D(128, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    b1_block2       = BatchNormalization()(block2)
    b1_block2       = Activation('elu')(block2)
    b1_block2       = AveragePooling2D((1, 8))(block2)
    b1_block2       = Dropout(dropoutRate)(block2)
