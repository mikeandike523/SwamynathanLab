# Copied and Adapted from Dr. Swamynathan Lab Goblet Cell Project 
# (D:\SwamynathanLab\ImageProcessingProjectLatestCodeWithGitVersionTracking\project\src\networks\u_net.py)

# U-net architecture adapted from https://blog.paperspace.com/unet-architecture-image-segmentation/

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, RandomFlip
import numpy as np
import tensorflow

BASE_FILTERS = 128

DROPOUT_RATE = 0.25

USE_RANDOM_FLIP = False

def convolve(input_tensor, filter_multiplier):
    input_tensor = Conv2D(BASE_FILTERS * filter_multiplier, 3, 1, 'same', 'channels_last')(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    input_tensor = Activation('relu')(input_tensor)
    input_tensor = Conv2D(BASE_FILTERS * filter_multiplier, 3, 1, 'same', 'channels_last')(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    input_tensor = Activation('relu')(input_tensor)
    return input_tensor


def encoder(input_tensor, filter_multiplier):
    skip_output = convolve(input_tensor, filter_multiplier)
    encoded_output = MaxPooling2D(strides=2, data_format='channels_last')(skip_output)
    return skip_output, encoded_output


def decoder(input_tensor, skip_output, filter_multiplier):
    upsampled = Conv2DTranspose(BASE_FILTERS * filter_multiplier, 2, 2, 'same', None, 'channels_last')(input_tensor)
    combined = Concatenate(axis=-1)([upsampled, skip_output])
    return convolve(combined, filter_multiplier)


def execute_u_net(input_tensor, SZ, _BASE_FILTERS=128):
    global BASE_FILTERS
    BASE_FILTERS = _BASE_FILTERS

    if USE_RANDOM_FLIP:
        input_tensor = RandomFlip('horizontal_and_vertical')(input_tensor)

    shp = np.asarray(tf.shape(input_tensor)._inferred_value[1:])
    assert tuple(shp) == (SZ, SZ, 3)

    s1, e1 = encoder(input_tensor, 1)
    s2, e2 = encoder(e1, 2)
    s3, e3 = encoder(e2, 4)
    s4, e4 = encoder(e3, 8)

    mid_layer = convolve(e4, 16)

    if DROPOUT_RATE is not None:
        mid_layer = Dropout(rate=DROPOUT_RATE)(mid_layer)

    d1 = decoder(mid_layer, s4, 8)
    d2 = decoder(d1, s3, 4)
    d3 = decoder(d2, s2, 2)
    d4 = decoder(d3, s1, 1)

    num_classes = 3

    classified_tensor = Conv2D(num_classes, 1, 1, 'same', 'channels_last', activation='sigmoid')(d4)

    classified_tensor = tf.expand_dims(classified_tensor, axis=-1)

    return classified_tensor


def UNetModel(SZ, _BASE_FILTERS=128):
    input_tensor = tensorflow.keras.layers.Input((SZ, SZ, 3))
    output_tensor = execute_u_net(input_tensor, SZ, _BASE_FILTERS)
    return tensorflow.keras.models.Model(input_tensor, output_tensor)


# New API
def UNet(size, nBaseFilters, useRandomFlip=False, dropoutRate=None):

    # Hacky
    global BASE_FILTERS
    global DROPOUT_RATE
    global USE_RANDOM_FLIP

    # Just to add flexibility to the API
    if dropoutRate == 0:
        dropoutRate = None

    BASE_FILTERS = nBaseFilters
    DROPOUT_RATE = dropoutRate

    input_tensor = tensorflow.keras.layers.Input((size, size, 3))
    output_tensor = execute_u_net(input_tensor, size, BASE_FILTERS)
    return tensorflow.keras.models.Model(input_tensor, output_tensor)