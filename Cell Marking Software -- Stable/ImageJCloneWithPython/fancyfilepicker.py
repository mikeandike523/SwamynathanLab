import os
import wx
from threading import Thread
import pickle
import cv2

DEFAULT_SCALE = 1.0

def get_last_start_folder():
    if os.path.exists("temp/last_start_folder.txt"):
        with open("temp/last_start_folder.txt","r") as fl:
            return fl.read()
    else:
        return os.getcwd()

def set_last_start_folder(path):
    with open("temp/last_start_folder.txt","w") as fl:
        fl.write(path)

class FancyFilePickerApplication:
    
    def __init__(self):
        self.app = wx.App()
        self.frame = wx.Frame(None,title="Fancy File Picker")
        self.build_ui()
    #@private
    def build_ui(self):

        self.vertical_sizer = wx.BoxSizer(wx.VERTICAL)
        self.control_panel = wx.Panel(self.frame,wx.ID_ANY)
        self.horizontal_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.control_panel.SetSizer(self.horizontal_sizer)
        self.frame.SetSizer(self.vertical_sizer) 
        self.dir_ctrl = wx.GenericDirCtrl(self.frame,wx.ID_ANY,wx.EmptyString,wx.DefaultPosition,wx.Size(400,300))
        self.vertical_sizer.Add(self.dir_ctrl,wx.SizerFlags().Expand())
        self.vertical_sizer.Add(self.control_panel,wx.SizerFlags().Expand())
        self.scale_factor_label = wx.StaticText(self.control_panel,wx.ID_ANY,"Scale Factor ".format(DEFAULT_SCALE))
        self.scale_factor_slider = wx.Slider(self.control_panel,wx.ID_ANY,value=int(DEFAULT_SCALE*100),minValue=0,maxValue=100)
        self.open_button = wx.Button(self.control_panel,wx.ID_ANY,"Open")
        self.horizontal_sizer.Add(self.scale_factor_label,wx.SizerFlags().Expand())
        self.horizontal_sizer.Add(self.scale_factor_slider,wx.SizerFlags().Expand())
        self.horizontal_sizer.Add(self.open_button,wx.SizerFlags().Expand())
        self.open_button.Bind(wx.EVT_BUTTON,lambda evt:self.submit())

        self.frame.Bind(wx.EVT_CLOSE,self.cancel)

        self.dir_ctrl.Bind(wx.EVT_DIRCTRL_FILEACTIVATED,lambda evt:self.submit())

        self.last_path = None

        if os.path.isfile("fancyfilepicker_lastpath.txt"):
            with open("fancyfilepicker_lastpath.txt","r") as fl:
                self.last_path = fl.read()

        if self.last_path is not None:
            self.dir_ctrl.ExpandPath(self.last_path)

    def open(self, file_picked_callback):
        self.file_picked_callback = file_picked_callback
        self.frame.Fit()
        self.frame.Center()
        self.frame.Show()
        self.frame.Raise()
        self.frame.ToggleWindowStyle(wx.STAY_ON_TOP)
        self.app.MainLoop()

    #@private
    def submit(self):
        filepath =self.dir_ctrl.GetFilePath()
        with open("fancyfilepicker_lastpath.txt","w") as fl:
            fl.write(filepath)
        self.file_picked_callback(filepath,self.scale_factor_slider.GetValue()/100)
        self.frame.Destroy()

    #@private
    def cancel(self,evt):
        self.file_picked_callback("",0)
        self.frame.Destroy()
