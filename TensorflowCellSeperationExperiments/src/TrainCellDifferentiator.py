#Attributions
# U-net architecture design adapted from https://blog.paperspace.com/unet-architecture-image-segmentation/
#       see u_net.py
# BatchProvider class concept adapted from https://blog.paperspace.com/unet-architecture-image-segmentation/
# Custom Callback design tutorial: https://www.tensorflow.org/guide/keras/custom_callback


#import necessary built in packages
import pickle
from this import d

# Import Necessary 3rd party Packages
import tensorflow as tf
import numpy as np
import random
import cv2
from scipy.spatial import KDTree
import PIL
import scipy.ndimage

# Import custom scripts
import image_processing as IP
import u_net as UNET
import utils as U

#seed rng
random.seed()
np.random.seed()




# Application Parameters
W=128
H=128
IO_PAIR_CIRCLE_MIN_RADIUS = 2
IO_PAIR_CIRCLE_MAX_RADIUS = 6
TARGET_RADIUS = 1
IO_PAIR_NUM_CIRCLES = 20
POISSON_DISK_DISTANCE = 4
POISSON_DISK_TRIES = 100
NUM_TRAINING_PAIRS = 25
NUM_VALIDATION_PAIRS = 10
BATCH_SIZE = 5
LEARNING_RATE = 100e-6
NUM_EPOCHS = 500
FRACTION_PERFECT_CIRCLE = 1.0
BLUR_RADIUS=1
BLUR_ITERS = 1
EPOCHS_PER_VIS = 5




# Create a function that will generate an input-output pair
def get_io_pair():

    target = np.zeros((H,W,3),dtype=np.uint8)
    supplied = np.zeros((H,W,3),dtype=np.uint8)

    locations = []

    for _ in range(IO_PAIR_NUM_CIRCLES):
        

        if len(locations) > 0:
            tree = KDTree(locations)
            
            x= random.randrange(0,W)
            y= random.randrange(0,H)

            d, _ = tree.query((x,y),1)

            tries = 0

            continue_flag = False

            while d < POISSON_DISK_DISTANCE:
                x= random.randrange(0,W)
                y= random.randrange(0,H)
                d, _ = tree.query((x,y),1)
                tries += 1
                if tries == POISSON_DISK_TRIES:
                    continue_flag = True

            if continue_flag:
                continue

        else:
            x= random.randrange(0,W)
            y= random.randrange(0,H)
        if np.random.uniform() < FRACTION_PERFECT_CIRCLE:
            r = random.randint(IO_PAIR_CIRCLE_MIN_RADIUS,IO_PAIR_CIRCLE_MAX_RADIUS)
            supplied = cv2.circle(supplied,(x,y),r,(255,255,255),-1)
        else:
            d1 = 2*random.randint(IO_PAIR_CIRCLE_MIN_RADIUS,IO_PAIR_CIRCLE_MAX_RADIUS)
            d2 = 2*random.randint(IO_PAIR_CIRCLE_MIN_RADIUS,IO_PAIR_CIRCLE_MAX_RADIUS)
            angle = np.random.uniform()*2.0*np.pi
            supplied = cv2.ellipse(supplied,(x,y),(d1,d2),angle,0,360,(255,255,255),-1)
        target = cv2.circle(target,(x,y),TARGET_RADIUS,(255,255,255),-1)

        locations.append((x,y))

    supplied_gs = np.where(supplied[:,:,0] > 0,1,0).astype(float)

    element= IP.circular_structuring_element(BLUR_RADIUS).astype(float)

    element/=np.sum(element)

    for _ in range(BLUR_ITERS):
        supplied_gs = np.clip(scipy.ndimage.convolve(supplied_gs,element,mode='constant',cval=0.0),0,1)

    supplied = IP.greyscale_plot_to_color_image(supplied_gs)

    return supplied, target

supplied, target = get_io_pair()

test_image = np.hstack((supplied,target))

IP.imshow(test_image)

model = UNET.UNetModel(128,128)

U.init_folder('temp/training_pairs')
U.init_folder('temp/validation_pairs')
U.init_folder('temp/mid_training_validation')

"""Generate Training Data"""
training_pair_filenames = []
validation_pair_filenames = []
for idx in range(NUM_TRAINING_PAIRS):
    pair = get_io_pair()
    pair_image = np.hstack(pair)
    PIL.Image.fromarray(pair_image).save(f"temp/training_pairs/{idx}.png")
    training_pair_filenames.append(f"temp/training_pairs/{idx}.png") 
for idx in range(NUM_VALIDATION_PAIRS):
    pair = get_io_pair()
    pair_image = np.hstack(pair)
    PIL.Image.fromarray(pair_image).save(f"temp/validation_pairs/{idx}.png")
    validation_pair_filenames.append(f"temp/validation_pairs/{idx}.png")  

np.random.shuffle(training_pair_filenames)
np.random.shuffle(validation_pair_filenames)

class BatchProvider(tf.keras.utils.Sequence):

    def __init__(self, batch_size, image_pair_filenames):
        self.batch_size = batch_size
        self.image_pair_filenames = image_pair_filenames

    def __len__(self):
        return len(self.image_pair_filenames) // self.batch_size

    def __getitem__(self,idx):
        X = []
        Y = []
        start = idx * self.batch_size
        end = start + self.batch_size
        for image_id in range(start, end):
            image_filename = self.image_pair_filenames[image_id]
            image_data = np.asarray(PIL.Image.open(image_filename))
            x = image_data[:,:W]
            y = image_data[:,W:]
            x = np.expand_dims(np.mean(x.astype(float),axis=2)/255,axis=2)
            y = np.expand_dims(np.mean(y.astype(float),axis=2)/255,axis=2)
            X.append(x)
            Y.append(y)
        X=np.array(X,dtype=float)
        Y=np.array(Y,dtype=float)
        return X,Y
        
training_generation = BatchProvider(BATCH_SIZE, training_pair_filenames)
validation_generation = BatchProvider(BATCH_SIZE, validation_pair_filenames)

#MTV: Mid Training Validation
MTV_batch_X = []
MTV_batch_Y = []

for _ in range(BATCH_SIZE):
    supplied, target = get_io_pair()
    MTV_batch_X.append(np.mean(supplied.astype(float)/255,axis=2))
    MTV_batch_Y.append(np.mean(target.astype(float)/255,axis=2))
MTV_batch_X = np.array(MTV_batch_X)
MTV_batch_Y = np.array(MTV_batch_Y)


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss="mean_squared_logarithmic_error",metrics=["mean_squared_logarithmic_error","mean_squared_error"])

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

    def on_epoch_end(self,epoch,logs=None):
        MTV_batch_predictedY = self.model.predict(MTV_batch_X)
        rows = []
        for idx in range(BATCH_SIZE):
            rows.append(np.hstack((
                IP.greyscale_plot_to_color_image(IP.rescale_array(MTV_batch_X[idx])),
                IP.greyscale_plot_to_color_image(IP.rescale_array(MTV_batch_Y[idx])),
                IP.greyscale_plot_to_color_image(IP.rescale_array(np.squeeze(MTV_batch_predictedY[idx]))),
            )))
        figure_pixels = np.vstack(rows)

        if self.epoch_number % EPOCHS_PER_VIS == 0:
            PIL.Image.fromarray(figure_pixels).save(f"temp/mid_training_validation/epoch_{self.epoch_number}.png")

        self.epoch_number+=1



callbacks = [tf.keras.callbacks.ModelCheckpoint("temp/segmentation.h5",save_best_only=True),ApplicationCallback()]

model.fit(training_generation,epochs=NUM_EPOCHS,validation_data=validation_generation,callbacks=callbacks)








