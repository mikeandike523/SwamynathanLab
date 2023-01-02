from core import setup
setup.setup()

from simplefilepicker import askopenfilename
from objexplore import explore
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gui',action="store_true")
args=parser.parse_args()

pickle_path = askopenfilename()

with open(pickle_path, 'rb') as fl:
    obj = pickle.load(fl)

if args.gui:    
    explore(obj)
else:
    from IPython import embed
    embed()
