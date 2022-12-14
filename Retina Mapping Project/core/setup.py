import sys
import os
from .path import normalize_path
from .sys import ensure_in_path

def setup():
    ensure_in_path(normalize_path(os.getcwd(), make_absolute=True))
    ensure_in_path(
        normalize_path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+"/TensorflowCellSeperationExperiments/src")
                   )