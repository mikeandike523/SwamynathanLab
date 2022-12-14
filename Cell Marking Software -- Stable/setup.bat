@echo off
echo "Setting up application..."
call %USERPROFILE%\Anaconda3\Scripts\Activate.bat base
powershell -ExecutionPolicy Bypass setup.ps1