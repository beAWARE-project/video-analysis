import socket
import sys
from threading import Lock, Thread
import time
import json
import urllib.request
import io
import os
import numpy as np
import requests
import skvideo.io
import video_analyzer
import datetime

#get lock object
lock = Lock()

#open logger
f = open('log.txt', 'a')

HOST = ''   # Symbolic name meaning all available interfaces
PORT = 9999 # Arbitrary non-privileged port

storage_link = 'http://object-store-app.eu-gb.mybluemix.net/objectStorage?file='

def load_video(path):
    #TODO: search metadata for timestamp
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

def download_from_storage(video_url):
    A = urllib.request.urlopen(video_url)
    data = A.read()
    vfile = open('temp.mp4', 'wb')
    vfile.write(data)
    vfile.close()
    fps, width, height, frames = load_video('temp.mp4')
    os.remove('temp.mp4')
    print("Printing video info: fps, width, height, frames.shape")
    print(fps, width, height, len(frames))
    return fps, width, height, frames

def process_video(fps, width, height, frames, timestamp, file_name):
    start = time.time()
    video_analyzer.analyze(frames, width, height, fps, file_name, timestamp)
    end = time.time()
    runtime_a = end-start
    start = time.time()
    bjson_output = open('./output/'+file_name+'_output.json', 'rb')
    bvid_output = open('./output/'+file_name+'_output.mp4', 'rb')
    save_to_storage(bvid_output, file_name+'_output.mp4')
    save_to_storage(bjson_output, file_name+'_output.json')
    bvid_output.close()
    bjson_output.close()
    end = time.time()
    runtime_u = end-start
    dict_to_send = {"message":{"vid_analyzed":storage_link+file_name+'_output.mp4', "vid_analysis":storage_link+file_name+'_output.json'}}
    bjson_links = json.dumps(dict_to_send).encode()
    return bjson_links, runtime_a, runtime_u
    
def save_to_storage(bobj, filename):
    r = requests.post(storage_link+filename, bobj)
    if not r.ok:
        print('Didn\'t happen')

def send_to_certh_hub(bjson_links, conn):
    conn.send(bjson_links)

def handle_message(bmsg, conn):
    msg = bmsg.decode()
    mydict = json.loads(msg)
    video_url = mydict['message']['URL']
    #timestamp = mydict['message']['startTimeUTC']
    timestamp = datetime.datetime.utcnow()
    start = time.time()
    fps, width, height, frames = download_from_storage(video_url)
    end = time.time()
    runtime_d = end - start
    #f.write("Download complete. Runtime: {0}\n".format(runtime))
    file_name = mydict['message']['URL'].split(sep='file=')[1].rsplit(sep='.', maxsplit=1)[0]
    bjson_links, runtime_a, runtime_u = process_video(fps, width, height, frames, timestamp, file_name)
    send_to_certh_hub(bjson_links, conn)
    #os.remove('./output/'+file_name+'_output.avi')
    #os.remove('./output/'+file_name+'_output.json')
    return runtime_d, runtime_a, runtime_u

#Function for handling connections. This will be used to create threads
def clientthread(conn):
    #global f
    lock.acquire()
    f = open('log.txt', 'a')
    while 1:
        bmsg = conn.recv(1024)
        msg = bmsg.decode()
        f.write("Hub says: "+msg+'\n')
        if(msg=="Msg from VA received"):
            #to_send = "Bye hub!"
            #conn.sendall(to_send.encode())
            break
        else:
            to_send = "Msg received from Hub"
            conn.sendall(to_send.encode())
            start = time.time()
    #try:
        run_d, run_a, run_u = handle_message(bmsg, conn)
        end = time.time()
        runtime = end - start
        f.write("Download complete. Runtime: {0}\n".format(run_d))
        f.write("Video analysis done. Runtime: {0}\n".format(run_a))
        f.write("Upload complete. Runtime: {0}\n".format(run_u))
        f.write("Message handling done. Runtime: {0}\n".format(runtime))
    #except:
        #print("A problem occured, please check the url link or the signal format")
        #f.write("Unknown error occured\n")
        #break

    conn.close()
    f.write('Connection closed\n')
    f.write(time.strftime('%X %x %Z')+'\n')
    f.close()
    blog = open('log.txt', 'rb')
    save_to_storage(blog, "video-analysis.log")
    lock.release()
    return

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
f.write('Socket created\n')
 
#Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    f.write('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1] + '\n')
    sys.exit()    
f.write('Socket bind complete\n')

#Start listening on socket
s.listen(10)
f.write('Socket now listening\n')
f.close()
t = []
#now keep talking with the client
while 1:
    #wait to accept a connection - blocking call
    #f.write('Waiting for a new connection...')
    conn, addr = s.accept()
    f = open('log.txt', 'a')
    f.write('Connected with ' + addr[0] + ':' + str(addr[1]) + '\n')
    f.close()
    #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
    t += [Thread(target=clientthread , args=(conn,))]
    t[-1].start()

s.close()
