# OSC Interface
# recieves OSC messages from the UI application,
# forwards generation parameters to generator wrapper,
# returns response from generator (generated file location) to UI app

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc import udp_client
from threading import Thread
from time import sleep

import mmm_ganGenerator as gg

print("TechnoGAN starting up - this might take a few seconds...")

# osc prefixes for messages to and from the generator server
CMD_PREFIX = "/gg/"
RETURN_PREFIX = "/ggret/"


class OSCClient(udp_client.SimpleUDPClient):
    def __init__(self, server="127.0.0.1", port=10101):
        udp_client.SimpleUDPClient.__init__(self, server, port)

    def send_msg(self, msg, *args):
        self.send_message(address=RETURN_PREFIX + str(msg), value=args)


class OSCServer(ThreadingOSCUDPServer):
    def __init__(self, server="127.0.0.1", port=10102, clients=frozenset()):
        dp = Dispatcher()
        dp.map(CMD_PREFIX + "*", self.handle_req)
        ThreadingOSCUDPServer.__init__(self, (server, port), dispatcher=dp)

    def handle_req(self, cmd: str, *params):
        # print(self, cmd, params)
        if cmd.startswith(CMD_PREFIX):
            cmd = cmd[len(CMD_PREFIX):]

            if cmd in gg.__dict__ and not cmd.startswith("_"):
                cmd = gg.__dict__[cmd]
                res_cmd, res_params, add_msg = cmd(params)
                OSCCLIENT.send_msg(res_cmd, res_params)
                if not add_msg == '':
                    OSCCLIENT.send_msg('msg', add_msg)
            else:
                print(f'{cmd} unrecognized command')
        else:
            print(f'osc msg: {cmd}\nnot starting with {CMD_PREFIX}, not my circus, but shouldn\'t get here anyway')


def _serve_daemon(target):  # helper to start in a daemon thread
    t = Thread(target=target.serve_forever)
    # t.setDaemon(True)
    t.daemon = True
    t.start()
    return t


def osc_startup():
    print("Starting OSC server...")
    s = OSCServer()
    _serve_daemon(s)
    c = OSCClient()
    c.send_msg('init', 'OSC Server starting...')
    print("OSC server started.")
    return c


def test_osc_single():
    server_addr = "127.0.0.1", 10102, 440
    osc_addr = "127.0.0.1", 10101
    s = OSCServer(*osc_addr, clients=[server_addr])
    t2 = _serve_daemon(s)


OSCCLIENT = osc_startup()

if __name__ == '__main__':
    while(True):
        sleep(10000)