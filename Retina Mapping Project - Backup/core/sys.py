import sys

def ensure_in_path(path):
    if path not in sys.path:
        sys.path.insert(0,path)