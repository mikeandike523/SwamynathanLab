# assuming cwd is .../src

import sys
import os
from pycheckpoint.checkpoint import CheckpointManager
import argparse
from tkinter import filedialog
import numpy as np
import PIL.Image

sys.path.insert(0,os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint")
args = parser.parse_args()

checkpoint = ""
if args.checkpoint:
    checkpoint = args.checkpoint

checkpoint_manager = CheckpointManager("./temp/mapping.py/checkpoints")

"""Begin project specific behaviour"""

if not checkpoint:
    checkpoint = "get_image"

def get_image(state):

    state.image_filepath = filedialog.askopenfilename()
    state.image = PIL.Image.open(state.image_filepath)

    return state

def get_spines(state):

    return state


if checkpoint == "get_image":

    checkpoint_manager.run_and_save_as(get_image, "get_image",None)

if checkpoint == "get_spines":

    checkpoint_manager.run_and_save_as(get_spines, "get_spines","get_image")

