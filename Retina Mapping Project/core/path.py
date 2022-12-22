import os
from pathlib import Path
import uuid
import shutil
import time

from .pythonic import handle_varargs_or_iterable

FS_POLL_TIME = 0.150 # seconds

class FilesystemItemType: # pythonic way of defining an enum
    FILE = 0
    DIRECTORY = 1


def normalize_path(path,make_absolute=False):
    
    if make_absolute:
        path = os.path.realpath(path)

    path=path.replace('\r\n','\n').strip('\n').strip('\t').strip(' ')
    path = path.replace('\\','/')
    path=path.rstrip('/')

    return path


def assert_existence_and_type(path, item_type: int):
    """! Void function --> Raises exception if file or directory does not exist"""
    
    if item_type == FilesystemItemType.FILE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File '{path}' does not exist.")
        if os.path.isdir(path):
            raise IsADirectoryError(f"Path '{path}' exists but is a directory.")

    if item_type == FilesystemItemType.DIRECTORY:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' does not exist.")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Path '{path}' exists but is not a directory.")

def blocking_unlink(path):
    assert_existence_and_type(path,FilesystemItemType.FILE)
    os.unlink(path)
    while os.path.isfile(path):
        time.sleep(FS_POLL_TIME)

def blocking_rmtree(path):
    assert_existence_and_type(path,FilesystemItemType.DIRECTORY)
    shutil.rmtree(path)
    while os.path.isdir(path):
        time.sleep(FS_POLL_TIME)


def clear_folder(path):
    assert_existence_and_type(path, FilesystemItemType.DIRECTORY)
    items = list(os.listdir(path))
    for item in items:
        if os.path.isfile(join_paths(path,item)):
            blocking_unlink(join_paths(path,item))
        if os.path.isdir(join_paths(path,item)):
            blocking_rmtree(join_paths(path,item))

def create_directory_with_parents(path):
    path = normalize_path(path,make_absolute=True)
    components = path.split('/')
    for i in range(len(components)):
        dirname = "/".join(components[:i+1])
        if os.path.isfile(os.path.normpath(dirname)):
            raise NotADirectoryError(f"Path {dirname} is a file.")
        if not os.path.exists(os.path.normpath(dirname)):
            os.mkdir(dirname)

def init_folder(path,clear=True):
    path = normalize_path(path,make_absolute=True)

    create_directory_with_parents(path)

    if clear:
        clear_folder(path)

def join_paths(*paths):
    paths = handle_varargs_or_iterable(paths)
    return "/".join(paths)

def strip_ext(path):

    return ".".join(path.split(".")[:-1])

def uuid_in(path):

    id_list = [strip_ext(fn) for fn in os.listdir(path)]
    
    id = uuid.uuid4()

    while id in id_list:
        id = uuid.uuid4()

    return str(id)

def remove_fs_significant_chars(path):
    # filesystem reservered characters: https://stackoverflow.com/a/31976060
    
    return path.replace("/","").replace("\\","").replace("|","").replace("*","").replace("?","").replace(":","").replace("<","").replace(">","").replace("\"","").replace("\'","")
