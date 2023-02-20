# compiles the generator into a windows console app with an icon in the notification bar
# this is the main file for freezing the generator app on Windows

import wx.adv
import wx
TRAY_TOOLTIP = 'TechnoGAN alpha' 
TRAY_ICON = 'icon.png' 

import win32gui
import win32.lib.win32con as win32con
import win32console

def create_menu_item(menu, label, func):
    item = wx.MenuItem(menu, -1, label)
    menu.Bind(wx.EVT_MENU, func, id=item.GetId())
    menu.Append(item)
    return item


class TaskBarIcon(wx.adv.TaskBarIcon):
    def __init__(self, frame):
        self.frame = frame
        super(TaskBarIcon, self).__init__()
        self.set_icon(TRAY_ICON)
        self.Bind(wx.adv.EVT_TASKBAR_LEFT_DOWN, self.on_left_down)

        self.console_window = win32console.GetConsoleWindow()

        hMenu = win32gui.GetSystemMenu(self.console_window, 0)
        if hMenu:
            win32gui.DeleteMenu(hMenu, win32con.SC_CLOSE, win32con.MF_BYCOMMAND)
        
        win32gui.ShowWindow(self.console_window, win32con.SW_HIDE)
        self.showing_console = False

    def CreatePopupMenu(self):
        menu = wx.Menu()
        create_menu_item(menu, 'Toggle console', self.toggle_console)
        menu.AppendSeparator()
        create_menu_item(menu, 'Close TechnoGAN Generator', self.on_exit)
        return menu

    def set_icon(self, path):
        icon = wx.Icon(path)
        self.SetIcon(icon, TRAY_TOOLTIP)

    def on_left_down(self, event):      
        print ('yo!')

    def toggle_console(self, event):
        if self.showing_console:
                win32gui.ShowWindow(self.console_window, win32con.SW_HIDE)
                self.showing_console = False
        else:
                win32gui.ShowWindow(self.console_window, win32con.SW_SHOW)
                self.showing_console = True


    def on_exit(self, event):
        wx.CallAfter(self.Destroy)
        self.frame.Close()

class App(wx.App):
    def OnInit(self):
        frame=wx.Frame(None)
        self.SetTopWindow(frame)
        TaskBarIcon(frame)
        return True

def main():
    import mmm_liveOSCServer
    app = App(False)
    app.MainLoop()


if __name__ == '__main__':
    main()