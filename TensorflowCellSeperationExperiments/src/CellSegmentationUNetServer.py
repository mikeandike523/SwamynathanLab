#Attributions
# U-net architecture design adapted from https://blog.paperspace.com/unet-architecture-image-segmentation/
#       see u_net.py
# BatchProvider class concept adapted from https://blog.paperspace.com/unet-architecture-image-segmentation/
# Custom Callback design tutorial: https://www.tensorflow.org/guide/keras/custom_callback


#import necessary built in packages
import pickle
import math
import base64

# Import Necessary 3rd party Packages
import tensorflow as tf
import numpy as np
import random
import cv2
from scipy.spatial import KDTree
import PIL
import scipy.ndimage

PIL.Image.MAX_IMAGE_PIXELS = None

# Import custom scripts
import image_processing as IP
import u_net as UNET
import utils as U

#seed rng
random.seed()
np.random.seed()

model = UNET.UNetModel(128,128)

"""Dont actually have to compile model when loading existing weights"""
#model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss="mean_squared_logarithmic_error",metrics=["mean_squared_logarithmic_error","mean_squared_error"])

U.init_folder("debug/panorama_snippet_results")

model.load_weights("/src/temp/segmentation.h5")

"""Set up flask server"""

from flask import Flask, request, Response
app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run():
    input_mask_bytes = request.form.get("input_mask_data")
    input_mask = np.frombuffer(input_mask_bytes, dtype=np.float32).reshape((128,128))
    input_batch = [np.expand_dims(input_mask,axis=2)]
    output = np.ravel(np.squeeze(model.predict(input_batch)).astype(np.float32))
    output.buffer.tobytes()

    return Response({"output_mask_data":base64.urlsafe_b64encode(output)},mimetype="application/json")

app.run('127.0.0.1',port=1738)