import os

import wx
import numpy as np

from core.io import imload

SIZE_MULTIPLER_W = 1.20
SIZE_MULTIPLER_H = 1.40


class ImageViewer(wx.Frame):

    def __init__(self, title="Image Viewer", init_pixels=None):

        if init_pixels is None:
            init_pixels = imload(os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "logo.webp"))
            title = "logo.webp"

        self.title = title

        self.pixels = init_pixels.copy()

        self.H, self.W, self.nchannels = self.pixels.shape  # IMAGE size NOT window size

        super().__init__(None, title=self.title, size=(
            int(SIZE_MULTIPLER_W*self.W), int(SIZE_MULTIPLER_H*self.H)
        ))

        self.inner_panel = wx.Panel(self, size=(self.W, self.H))

        self.inner_panel.SetBackgroundColour((0, 0, 255))

        # no need to add sizers to self

        centering_sizer = wx.BoxSizer()

        centering_sizer.Add(self.inner_panel, wx.ALIGN_CENTER)

        self.SetSizer(centering_sizer)

        self.inner_panel.Bind(wx.EVT_PAINT, self.OnPanelPaint)

        self.inner_panel.Bind(wx.EVT_IDLE, self.OnPanelIdle)

    def Show(self, center_on_screen=True):  # Follow wxPython naming conventions
        if center_on_screen:
            super().Centre()
        super().Show()

    def OnPanelPaint(self, e):
        dc = wx.PaintDC(self.inner_panel)
        
        bmp = wx.BitmapFromBufferRGBA(self.W, self.H, np.dstack((
            *[self.pixels[:, :, c] for c in range(3)],
            np.zeros(
                (self.H, self.W), np.uint8)+255
            )))
        
        dc.DrawBitmap(bmp,0, 0)

    def OnPanelIdle(self, e):
        self.inner_panel.Refresh()
        e.RequestMore()
