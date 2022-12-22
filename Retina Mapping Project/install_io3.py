import os

path = "D:\\SwamynathanLab\\io3\\target\\wheels"

for file in os.listdir(path):
    os.system(f"python -m pip install --force-reinstall {os.path.normpath(os.path.join(path,file))}")
    break

