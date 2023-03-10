# implements keyboard shortcuts during training
# ctrl+shift+s - save training progress after current epoch is finished
# ctrl+shift+p - pause/resume training without halting the script
#
# BUG: for some reason, occasionally python is unable to regain focus, so keybard interaction stops working

import keyboard


class keyboardUI():

    def savenext(self):
        self.SAVENEXT = not self.SAVENEXT
        if self.SAVENEXT:
            print('saving checkpoint after current step, ctrl-shift-s to cancel...\n')
        else:
            print('saving canceled\n')

    def donesaving(self):
        self.SAVENEXT = False

    def pause(self):
        self.PAUSED = not self.PAUSED
        if self.PAUSED:
            print('pausing training, ctrl-shift-p to continue...')
        else:
            print('moving on...')

    def __init__(self):
        self.PAUSED = False
        self.SAVENEXT = False
        keyboard.add_hotkey("ctrl+shift+s", self.savenext)
        keyboard.add_hotkey("ctrl+shift+p", self.pause)

