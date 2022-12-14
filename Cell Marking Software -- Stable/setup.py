import os
import re

def normalize_path(path):
    path = path.replace("\\","/")
    path = path.rstrip("/")
    path = re.sub("\/+","/",path)
    return path

print("Creating a shortcut on desktop...")

traditional_desktop_path = normalize_path(os.getenv('USERPROFILE')+"Desktop")
onedrive_desktop_path = normalize_path(os.getenv('USERPROFILE')+"/OneDrive/Desktop")

if not os.path.isdir(traditional_desktop_path) and not os.path.isdir(onedrive_desktop_path):
    print("Cannot finish installation: no desktop folder was found.")
    exit()

desktop_path = ""

if os.path.isdir(traditional_desktop_path):
    desktop_path = traditional_desktop_path

if os.path.isdir(onedrive_desktop_path):
    desktop_path = onedrive_desktop_path

project_folder = normalize_path(os.path.realpath(os.path.dirname(__file__))).replace("/","\\")

bat_file_source= f"""
cd /D \"{project_folder}\"
call run.bat
"""

with open(desktop_path+"/Mark Cells.bat","w") as fl:
    fl.write(bat_file_source)