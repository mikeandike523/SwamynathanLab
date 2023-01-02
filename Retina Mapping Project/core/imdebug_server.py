# Adapted from Swamynathan Lab "cellmark3" project: D:\SwamynathanLab\cellmark3\core\file_picker.py

import wx 
import os
from time import sleep
from core.path import init_folder
from multiprocessing import Process, Pipe
import numpy as np
from threading import Thread

PORT = 9001

def get_placeholder():

    return np.clip(255*np.random.random((300,300,3)),0,255).astype(np.uint8)

class ImDebugFrame(wx.Frame):
    
    def __init__(self, image, *args, **kwargs):
        
        screen_size = wx.DisplaySize()  
        
        kwargs["title"] = "ImDebugApp"
        
        kwargs["size"] = (int(screen_size[0]/2),int(screen_size[1]/2))
        
        super().__init__(None, *args, **kwargs)

        wxImage = wx.Image(image.shape[1], image.shape[0])

        wxImage.SetData(image.tobytes())

        self.sbmp = wx.StaticBitmap(self, wx.ID_ANY, bitmap=wx.Bitmap(wxImage))

        self.set_image(image)
        
    def set_image(self, image):

        wxImage = wx.Image(image.shape[1], image.shape[0])

        wxImage.SetData(image.tobytes())

        self.sbmp.SetBitmap(wx.Bitmap(wxImage))

        self.Fit()

        self.Center()

def server(child_conn, initial_image):
        app=wx.App() 
        frame = ImDebugFrame(initial_image)
        def recv_thread():

            while True:
                try:
                    # Receive data from the client
                    data = child_conn.recv()
                    # Print the received data
                    
                    frame.set_image(data)

                except EOFError:
                    # Connection closed, exit the loop
                    break

        frame.ToggleWindowStyle(wx.STAY_ON_TOP)
        frame.Show()
        frame.Centre()

        Thread(target=recv_thread, daemon=True).start()

        app.MainLoop()

class ImDebugServer:

    def __init__(self, initial_image = get_placeholder()):

        self.parent_conn, self.child_conn = Pipe()
     
        self.process = Process(target=server, daemon=True, args=(self.child_conn,initial_image))

    def start(self):

        self.process.start()

    def imshow(self, image):
        self.parent_conn.send(image)

    
        
