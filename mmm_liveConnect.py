import socket
import time


def make_osc_string(s):
    for i in range(4 - len(s) % 4):
        s = f'{s}\0'
    s = f'{s},\0\0\0'
    s = s.encode('utf-8')
    return s

def strip_osc_string(s):
    return s.decode('utf-8').rstrip('\0')[:-1].rstrip('\0')

UDP_IP = '127.0.0.1'
UDP_SENDPORT = 10101
UDP_RECPORT = 10102
MESSAGE = b'Hello, World!'


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_RECPORT))

msg = f'python sender started at {time.time()}'
print(make_osc_string(msg))
sock.sendto(make_osc_string(msg), (UDP_IP, UDP_SENDPORT))
#sock.send(MESSAGE)


while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    data = strip_osc_string(data)
    print(data, addr)

    answer = str(f"returning {data} at {time.time()}")
    sock.sendto(make_osc_string(answer), (UDP_IP, UDP_SENDPORT))
    print ("received message:", data)
    print ("returning message:", answer)