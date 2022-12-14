import wx

def askopenfilename(wildcard="*"):
    # --- Adapted from https://docs.wxpython.org/wx.FileDialog.html
    app=wx.App()
    with wx.FileDialog(None, "Open Saved Annotations", wildcard=wildcard,
                       style=wx.FD_OPEN) as fileDialog:
        if fileDialog.ShowModal() == wx.ID_CANCEL:
            return None
        pathname = fileDialog.GetPath()
        return pathname
    # ---