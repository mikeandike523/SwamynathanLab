# Adapted from Swamynathan Lab "cellmark3" project: D:\SwamynathanLab\cellmark3\core\file_picker.py

import wx 
import os

from time import sleep

from core.path import init_folder

class ResultCarrier:
    """A box that contains a result and whether it exists.
    Helps differentiate meaningful None values from nonexistent values
    Can only be set once
    """
    
    def __init__(self):
        self.result = None
        self.has_result = False
        
    def send(self,result):
        if not self.has_result:
            self.has_result = True
            self.result = result
        else:
            raise Exception("ResultCarrier has already been set")

class FilePicker(wx.Frame):
    
    def __init__(self, title, init_path, result_carrier, *args, **kwargs):
        
        screen_size = wx.DisplaySize()  
        
        kwargs["title"] = title
        
        kwargs["size"] = (int(screen_size[0]/2),int(screen_size[1]/2))
        
        super().__init__(None, *args, **kwargs)
        
        self.init_path = init_path
        self.result_carrier = result_carrier
        
        self.dir_ctrl = wx.GenericDirCtrl(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.dir_ctrl,wx.SizerFlags().Expand())
        
        if init_path:
            self.dir_ctrl.SetPath(init_path)
            self.dir_ctrl.ExpandPath(init_path)  
            
        self.dir_ctrl.Bind(wx.EVT_DIRCTRL_FILEACTIVATED, self.OnFileActivated)
        self.Bind(wx.EVT_CLOSE,self.OnClose)
        
    def OnFileActivated(self, event):
        self.result_carrier.send(self.dir_ctrl.GetPath())
        self.Destroy()
    
    def OnClose(self, event):
        self.result_carrier.send("")
        self.Destroy()

    
def askopenfilename(title="File Picker"):
    init_folder("temp", clear=False)
    app=wx.App()
    lastPath = None
    if os.path.exists("temp/lastPath.txt"):
        with open("temp/lastPath.txt", "r") as fl:
            lastPath = fl.read()
    result_carrier = ResultCarrier()  
    file_picker = FilePicker(title,lastPath,result_carrier)
    file_picker.ToggleWindowStyle(wx.STAY_ON_TOP)
    file_picker.Show()
    file_picker.Centre()
    app.MainLoop()
    if(result_carrier.result):
        lastPath = result_carrier.result
        with open("temp/lastPath.txt", "w") as fl:
            fl.write(lastPath)
    return result_carrier.result
    
