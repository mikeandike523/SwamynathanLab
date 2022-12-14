call C:\Users\micha\anaconda3\Scripts\activate.bat swamynathan_lab
for /R %%i in (image_trioxide\target\wheels\*.whl) DO python -m pip install --force-reinstall %%i
pause