from Xlib.display import Display
from Xlib.ext import xinput
import fcntl

class Handler:
    def __init__(self, display):
        self.enabled = True
        self.display = display

    def handle(self, event):
        if event.data['detail'] == 76 and event.data['mods']['base_mods'] == 4:
            if self.enabled:
                self.display.grab_server()
            else:
                self.display.ungrab_server()
            self.enabled = not self.enabled

display = Display()
try:
    handler = Handler(display)
    screen = display.screen()
    screen.root.xinput_select_events([
        (xinput.AllDevices, xinput.KeyPressMask),
    ])
    while True:
        event = display.next_event()
        handler.handle(event)
finally:
    display.close()