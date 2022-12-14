from core import setup
setup.setup()

import os
import random

import numpy as np
import PIL.Image
import cv2

import tensorflow as tf

import u_net as UNET
import utils as U
import image_processing as IP

from core.database import Database
from core.path import join_paths, init_folder


# Application Parameters
TRAINING_DATA_PATH = "output/training_pairs"
SZ = 64
BASE_FILTERS = 256
USE_RANDOM_FLIP = True
DROPOUT_RATE = None # Do not include dropout layer
NUM_TRAIN = 200
NUM_VAL = 50
BATCH_SIZE = 2
NUM_VIS = 5
EPOCHS = 150
EPOCHS_PER_VIS=2
LEARNING_RATE = 100e-6

# App Setup
init_folder("output",clear=False)

# helper functions

def toSZandrescaled(image):
    return (cv2.resize(image.copy(),dsize=(SZ,SZ),interpolation=cv2.INTER_CUBIC)).astype(float)/255

# Get filenames for training and validation data
all_filenames = [fn for fn in list(os.listdir(TRAINING_DATA_PATH)) if fn.endswith('.png')]

# Collect filenames for training and validation pairs
training_filenames = []
validation_filenames = []
visual_validation_filenames = []

for _ in range(NUM_TRAIN):
    random_idx = random.randrange(0,len(all_filenames))
    training_filenames.append(all_filenames.pop(random_idx))

for _ in range(NUM_VAL):
    random_idx = random.randrange(0,len(all_filenames))
    validation_filenames.append(all_filenames.pop(random_idx))

for _ in range(NUM_VIS):
    random_idx = random.randrange(0,len(all_filenames))
    visual_validation_filenames.append(all_filenames.pop(random_idx))

class BatchProvider(tf.keras.utils.Sequence):

    def __init__(self, name, pair_filenames):

        self.name = name
        self.database = Database(name,"cache/BatchProvider")
        self.database.reset() 
        self.pair_filenames = pair_filenames

        self.__load_images()

    def __load_images(self):
        
        U.dprint("Loading and serializing images...")

        pair_filenames = self.pair_filenames

        num_filenames = len(pair_filenames)

        for idx, pair_filename in enumerate(pair_filenames):

            print(f"File {idx+1} of {num_filenames}...")

            pixels = IP.imload(join_paths(TRAINING_DATA_PATH,pair_filename))

            H, twoW = pixels.shape[:2]

            input_pixels, output_pixels = toSZandrescaled(pixels[:,:twoW//2]), toSZandrescaled(pixels[:,twoW//2:])

            self.database.set_variable(f"input_pixels/{pair_filename}",input_pixels,direct=False)

            self.database.set_variable(f"output_pixels/{pair_filename}",output_pixels,direct=False)

    def __len__(self):

        return len(self.pair_filenames) // BATCH_SIZE

    def __getitem__(self,batch_idx):
        X=[]
        Y=[]
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        filename_list = self.pair_filenames[start_idx:end_idx]
        for filename in filename_list:
            X.append(self.database.get_variable(f"input_pixels/{filename}"))
            Y.append(self.database.get_variable(f"output_pixels/{filename}"))
        return np.array(X, float), np.array(Y, float)

train_gen = BatchProvider("train_gen", training_filenames)
val_gen = BatchProvider("val_gen",validation_filenames)
vis_val_gen = BatchProvider("vis_val_gen",visual_validation_filenames)

class ApplicationCallback(tf.keras.callbacks.Callback):
    """ *class* ApplicationCallback
    
    This callback will house all mid-training monitoring, including running images through the network as well as handling logging
    @TODO:
    * Log training and validation loss, as well as training and validation mean_squared_error
    Track the evolution of network output on a subset of the validation images
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_number = 0
        init_folder("output/mid_training_validation")

    def on_epoch_end(self,epoch,logs=None):

        MTV_batch_X, MTV_batch_Y = vis_val_gen[0]

        MTV_batch_predictedY = self.model.predict(MTV_batch_X)
        rows = []
        for idx in range(BATCH_SIZE):
            rows.append(np.hstack((
                IP.image_1p0_to_255(MTV_batch_X[idx]),
                IP.image_1p0_to_255( MTV_batch_Y[idx]),
                IP.image_1p0_to_255(np.squeeze(MTV_batch_predictedY[idx])),
            )))
        figure_pixels = np.vstack(rows)

        if self.epoch_number % EPOCHS_PER_VIS == 0:
           PIL.Image.fromarray(figure_pixels).save(f"output/mid_training_validation/epoch_{self.epoch_number}.png")

        self.epoch_number+=1

model = UNET.UNet(SZ,BASE_FILTERS,USE_RANDOM_FLIP,DROPOUT_RATE) # Size, Base-Filters

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),loss='mean_squared_logarithmic_error',metrics=["mean_squared_logarithmic_error","mean_squared_error"])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("output/segmentation.h5"),ApplicationCallback()
]

model.fit(train_gen,epochs=EPOCHS,validation_data=val_gen,callbacks=callbacks)