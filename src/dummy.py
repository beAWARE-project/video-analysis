import socket
import sys
from _thread import *
import time
import json
import urllib.request
from PIL import Image
import io
import os
import numpy as np
import requests
import skvideo.io

cwd = os.getcwd()
print(cwd)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
 
#Bind socket to local host and port
try:
    s.bind(('', 9999))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()
     
print('Socket bind complete')
 
#Start listening on socket
s.listen(10)
print('Socket now listening') 

while 1:
    #wait to accept a connection - blocking call
    print('Waiting for a new connection...')
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
     
    #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
    start_new_thread(clientthread ,(conn,))
 
s.close()
