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
 
HOST = ''   # Symbolic name meaning all available interfaces
PORT = 9999 # Arbitrary non-privileged port

storage_link = 'http://object-store-app.eu-gb.mybluemix.net/objectStorage?file='

def download_from_storage(video_url):
    A = urllib.request.urlopen(video_url)
    data = A.read()
    file = open('test.mp4', 'wb')
    file.write(data)
    fps, width, height, frames = load_video('test.mp4')
    print("Printing video info: fps, width, height, frames.shape")
    print(fps, width, height, len(frames))
    return fps, width, height, frames

def load_video(path):
    PATH_TO_VIDEO = path
    metadata = skvideo.io.ffprobe(PATH_TO_VIDEO)
    frames = []
    fps = metadata['video']['@r_frame_rate']
    fps = fps.split(sep='/')
    fps = int(fps[0])/int(fps[1])
    num_of_frames = int(metadata['video']['@nb_frames'])
    width = int(metadata['video']['@width'])
    height = int(metadata['video']['@height'])
    sequence = skvideo.io.vreader(PATH_TO_VIDEO)
    for f in sequence:
        frames += [f]
    return fps, width, height, frames


def process_video(fps, width, height, frames, timestamp, file_name):
    #do the analysis
    #here we just open the dummy files
    bvid_output = open('test_output.mp4', 'rb')
    bjson_output = open('test_output.json', 'rb')
    save_to_storage(bvid_output, file_name+'_output.mp4')
    save_to_storage(bjson_output, file_name+'_output.json')
    dict_to_send = {"message":{"vid_analyzed":storage_link+file_name+'_output.mp4', "vid_analysis":storage_link+file_name+'_output.json'}}
    bjson_links = json.dumps(dict_to_send).encode()
    return bjson_links
    
def save_to_storage(bobj, filename):
    #Upload to storage
    r = requests.post(storage_link+filename, bobj)
    if not r.ok:
        print('Didn\'t happen')

def send_to_certh_hub(bjson_links, conn):
    conn.send(bjson_links)

def handle_message(bmsg, conn):
    #time.sleep(10)
    msg = bmsg.decode()
    mydict = json.loads(msg)
    video_url = mydict['message']['URL']
    timestamp = mydict['message']['startTimeUTC']
    fps, width, height, frames = download_from_storage(video_url)
    file_name = mydict['message']['URL'].split(sep='file=')[1].split(sep='.')[0]
    bjson_links = process_video(fps, width, height, frames, timestamp, file_name)
    send_to_certh_hub(bjson_links, conn)
    print("Video handling done")
    os.remove('test.mp4')
    return

#Function for handling connections. This will be used to create threads
def clientthread(conn):
    while 1:
        bmsg = conn.recv(1024)
        msg = bmsg.decode()
        print("Hub says: ",msg)
        if(msg=="Msg from VA received"):
            #to_send = "Bye hub!"
            #conn.sendall(to_send.encode())
            break
        else:
            to_send = "Msg received from Hub"
            conn.sendall(to_send.encode())
            handle_message(bmsg, conn)
    conn.close()
    print('Connection closed')
    return

print('Hello World')
print('Hello World')
print('Hello World')
print('Hello World')
print('Hello World')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
 
#Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()
     
print('Socket bind complete')
 
#Start listening on socket
s.listen(10)
print('Socket now listening')
#now keep talking with the client
while 1:
    #wait to accept a connection - blocking call
    print('Waiting for a new connection...')
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
     
    #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
    start_new_thread(clientthread ,(conn,))
 
s.close()
