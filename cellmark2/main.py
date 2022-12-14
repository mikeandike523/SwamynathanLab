import wx

from core.io import imload
from core import image_ops
from gui.pygame_based.image_viewer import ImageViewer

# from gui.image_viewer import ImageViewer

# app = wx.App()

# image_viewer = ImageViewer()

# image_viewer.Show()

# app.MainLoop()

image_viewer = ImageViewer()

image_viewer.start()
