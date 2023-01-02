from core import setup
setup.setup()

import utils as U

filepath = U.askopenfilename()

import pickle

with open(filepath, 'rb') as fl:
    whatsinthere = pickle.load(fl)

from objexplore import explore

explore(whatsinthere)