import wx

def askopenfilename(wildcard="*"):
    # --- Adapted from https://docs.wxpython.org/wx.FileDialog.html
    app=wx.App()
    with wx.FileDialog(None, "Open Saved Annotations", wildcard=wildcard,
                       style=wx.FD_OPEN | wx.STAY_ON_TOP) as fileDialog:
        fileDialog.Raise()
        fileDialog.Maximize()
        if fileDialog.ShowModal() == wx.ID_CANCEL:
            return None
        pathname = fileDialog.GetPath()
        return pathname
    # ---