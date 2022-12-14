@echo off
echo "Running application..."
call %USERPROFILE%\Anaconda3\Scripts\Activate.bat cellmarkingsoftware
cd ImageJCloneWithPython
python main.py