import sys
import os
from .path import normalize_path
from .sys import ensure_in_path

def setup():
    ensure_in_path(normalize_path(os.getcwd(), make_absolute=True))

    file = os.path.realpath(__file__)

    ensure_in_path(
        normalize_path(os.path.dirname(os.path.dirname(os.path.dirname(file)))+"/TensorflowCellSeperationExperiments/src")
                   )
    os.chdir(os.path.dirname(os.path.dirname(__file__)))