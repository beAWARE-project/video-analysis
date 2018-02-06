
# coding: utf-8

# # v01
# - created 06-02-18
# - final version for the flood test
# - processes a single video locally stored
# - stores locally an output image and json with the same filename as the video
# - currently using FasterRCNN trained on COCO model
# - place in /usr/src/listener/
# - there are internal parameters

# # Import libs and model

# In[ ]:

import datetime
import numpy as np
import os
import six.moves.urllib as urllib
import tensorflow as tf
import pandas
from collections import defaultdict
#from io import StringIO
import time
import KCF
import cv2
from PIL import Image
from tqdm import tqdm
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import skvideo.io
import json

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './model/label_map.pbtxt'
NUM_CLASSES = 90
        
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#category_index.update({0: {'id': 0, 'name': 'Detection Lost'}})

print('Analyzer imported')

#PARAMETERS
confidence_drop = 0.6 #bellow this confidence score boxes will be dropped
upper_size_thres = 0.9 #above this size (percentage of screen) boxes will be dropped

def get_iou(bbb1, bbb2):
    
    bb1 = [bbb1[0], bbb1[0]+bbb1[2], bbb1[1], bbb1[1]+bbb1[3]]
    bb2 = [bbb2[0], bbb2[0]+bbb2[2], bbb2[1], bbb2[1]+bbb2[3]]
    
    assert bb1[0] < bb1[1]
    assert bb1[2] < bb1[3]
    assert bb2[0] < bb2[1]
    assert bb2[2] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def print_track(boundingbox, width, height, frame, idx, col, cus):
    cv2.rectangle(frame,(boundingbox[0],boundingbox[1]),
                  (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]),
                  col, 1)                       
    cv2.putText(frame,
                (str)(idx),
                ((int)(boundingbox[0]+boundingbox[2]/2), (int)(boundingbox[1]+boundingbox[3]/2)),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,51), 2)
    return

def box_trans(box, width, height): #transform from (ymin, xmin, ymax, xmax)[norm] to (left, top, width, height)[non-norm]
    box = [box[0] * height, box[1] * width, box[2] * height, box[3] * width]
    box = list(map(int, box))
    box = [box[1], box[0], abs(box[3]-box[1]), abs(box[2]-box[0])]
    return box

def rev_box_trans(box_list): #transform from (left, top, width, height)[non-norm] to (ymin, xmin, ymax, xmax)[non-norm]
    res = []
    for box in box_list:
        temp = [box[1], box[0], box[3]+box[1], box[2]+box[0]]
        res = res + [temp]
    return res

def to_norm(box_list, height, width): #transform from (left, top, width, height)[non-norm] to (ymin, xmin, ymax, xmax)[norm]
    res = []
    for box in box_list:
        temp = [box[1]/height, box[0]/width, (box[3]+box[1])/height, (box[2]+box[0])/width]
        res = res + [temp]
    return res

def export_json(frameList, width, height, fps, file_name, timestamp):
    target_list = []
    for idf, frame in enumerate(frameList):
        for target in frame:
            if target[0] not in target_list:
                target_list += [target[0]]
    target_exists = []
    for i in target_list:
        target_exists += [[]]
    for idf, frame in enumerate(frameList):
        for j, idt in enumerate(target_list):
            if idt in [target[0] for target in frameList[idf]]:
                target_exists[j] += [True]
            else:
                target_exists[j] += [False]

    existence_df = pandas.DataFrame(target_exists, index=target_list, columns=range(len(frameList)))
    existence_df = existence_df.transpose()
    extra_frame = pandas.DataFrame([False] * len(target_list)).transpose()
    extra_frame.index = [len(frameList)]
    extra_frame.columns = target_list
    existence_df = pandas.concat([existence_df, extra_frame])
    begins = existence_df.idxmax()
    ends = pandas.Series([existence_df[i][begins[i]:].idxmin()-1 for i in existence_df], index=target_list)
    targets_df = pandas.concat([begins, ends], axis=1)
    targets_df.columns = ['start_frame', 'end_frame']

    targetList = []
    for idt in target_list:
        idf = targets_df['start_frame'].loc[idt]
        target_single = []
        for frame in frameList[targets_df['start_frame'].loc[idt]:targets_df['end_frame'].loc[idt]+1]:
            thisframe = frame[[idx for idx, target in enumerate(frame) if(target[0]==idt)][0]]
            thisframe[5] = idf
            idf += 1
            target_single += [thisframe]
        targetList += [target_single]

    all_targets = []
    for target in targetList:
        frame_dict = []
        for frame in target:
            frame_dict += [{"frame_num":frame[5].__int__(),
                            "confidence":float("{0:.3f}".format(frame[2])),
                            "left":frame[3][0],
                            "top":frame[3][1], 
                            "width":frame[3][2], 
                            "height":frame[3][3]}]
        risk = np.random.random_sample()
        target_dict = {"id":target[0][0],
                       "type":category_index[int(target[0][1])]['name'], 
                       "starts":str(timestamp+datetime.timedelta(seconds=target[0][5]/fps)),
                       "ends":str(timestamp+datetime.timedelta(seconds=target[len(target)-1][5]/fps)),
                       "risk":risk,
                       "trajectory":frame_dict}
        all_targets += [target_dict]

    son = {'sequence':{"name":file_name,
                       "width":width,
                       "height":height,
                       "fps":float("{0:.3f}".format(fps)),
                       "crisis_type": "flood",
                       "crisis_level": "high",
                       "targets":all_targets}}

    with open('./output/'+file_name+'_output.json', 'w') as outfile:
        json.dump(son, outfile, sort_keys=False, indent=1)
    outfile.close()

def analyze(frames, width, height, fps, file_name, timestamp):
    start = time.time()
    tracks = []
    tracked_boxes = []
    frameList = []
    buffer = []
    #PARAMETERS
    video = True
    detection_rate = 3 #detection every this number of frames
    dlost_thres = 3 #delete track if detection lost for this number of frames
    reset_thres = 15 #clear the screen from tracked boxes every this number of frames
    confidence_drop = 0.6 #bellow this confidence score boxes will be dropped
    upper_size_thres = 0.9 #above this size (percentage of screen) boxes will be dropped
    save_iou = 0.33 #if you find an iou between current box and tracked boxes above that threshold don't create a new id
    rectify_iou = 0.7 #if the max iou between the tracked box and a detection box is bellow that rectify the tracked box
    merge_iou = 0.8 #if you find tracked boxes with iou score above that delete the newest track (merge)
   

    with tf.device('/gpu:0'):
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                #image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes_g = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores_g = detection_graph.get_tensor_by_name('detection_scores:0')
                classes_g = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections_g = detection_graph.get_tensor_by_name('num_detections:0')
                for idf, f in enumerate(tqdm(frames)):
                    #frame = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    frame_np = f
                    write_on = frame_np.copy()
                    height = frame_np.shape[0]
                    width = frame_np.shape[1]
                    # Actual detection.
                    if(idf%detection_rate == 0):
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        frame_np_expanded = np.expand_dims(frame_np, axis=0)
                        (boxes, scores, classes, num_detections) = sess.run(
                            [boxes_g, scores_g, classes_g, num_detections_g],
                            feed_dict={image_tensor: frame_np_expanded})
                        to_del=()
            
                        #delete unwanted boxes (too large, non car labels, etc)
                        for i in range(boxes.shape[1]):
                            if((np.abs(boxes[0][i][0]-boxes[0][i][2])*np.abs(boxes[0][i][1]-boxes[0][i][3]) >= upper_size_thres) or 
                               (scores[0][i] < confidence_drop) or
                               (classes[0][i] not in [1, 2, 3, 4, 6, 8, 9])):
                                to_del = to_del + (i,)
                                num_detections[0] = num_detections[0] - 1
                        boxes = np.delete(boxes, to_del, axis=1)
                        scores = np.delete(scores, to_del, axis=1)
                        classes = np.delete(classes, to_del, axis=1)
                        boxes_t=[] #initiallize transformed boxes storage
            
                        # Tracking
                        if(idf > 0 and idf%reset_thres == 0): #reset boxes every reset_thres_th frame
                            for idx, track in enumerate(tracks):
                                if(tracks[idx]):
                                    boundingbox = track.update(frame_np)
                                    tracked_boxes[idx] = [boundingbox]
                                else:
                                    tracked_boxes[idx] = []
                
                        for bid, box in enumerate(np.squeeze(boxes, axis=0)): #transform detected boxes and search through
                            box = box_trans(box, width, height)
                            boxes_t = boxes_t + [box]
                            #save new tracks
                    
                            save_flag = True
                            for sbox in sum(tracked_boxes, []): #check all tracked boxes
                                if(sbox): #if the sbox exists
                                    if(get_iou(sbox, box) > save_iou): #if you find an iou above that threshold don't save a new box
                                        save_flag = False    
                            if (save_flag == True): #initiallize a new tracker (you found a box with no iou above the threshold)
                                tracker = KCF.kcftracker(False, True, True, False)  # hog, fixed_window, multiscale, lab
                                tracks = tracks + [tracker]
                                tracked_boxes = tracked_boxes + [[box]]
                                tracker.init(box, frame_np)
            
                    frameList = frameList + [[]]            
                    for idx, track in enumerate(tracks):
                        if (tracks[idx]):
                                boundingbox = track.update(frame_np)
                                if(idf%detection_rate == 0): #every 3rd frame
                                    max_iou = 0
                                    for idb, ibox in enumerate(boxes_t):
                                        iou = get_iou(ibox, boundingbox)
                                        if(iou >= max_iou):
                                            max_iou_id = idb
                                            max_iou = iou
                                    if(max_iou > rectify_iou):
                                        #keep
                                        tracked_boxes[idx] = tracked_boxes[idx] + [boundingbox]
                                        #frameList append
                                        frameList[idf] = frameList[idf] + [[idx, 
                                                                           np.squeeze(classes, axis=0)[max_iou_id], 
                                                                           np.squeeze(scores, axis=0)[max_iou_id], 
                                                                           boundingbox, 
                                                                           boxes_t[max_iou_id],
                                                                           1]]
                                    elif(max_iou <= rectify_iou and max_iou > 0.1):
                                        #rect
                                        track.init(boxes_t[max_iou_id], frame_np)
                                        boundingbox = track.update(frame_np)
                                        tracked_boxes[idx] = tracked_boxes[idx] + [boundingbox]
                                        frameList[idf] = frameList[idf] + [[idx, 
                                                                           np.squeeze(classes, axis=0)[max_iou_id], 
                                                                           np.squeeze(scores, axis=0)[max_iou_id], 
                                                                           boundingbox, 
                                                                           boxes_t[max_iou_id],
                                                                           1]]
                                    else:
                                        tracked_boxes[idx] = tracked_boxes[idx] + [boundingbox]
                                        counter = 0
                                        if(idf >= detection_rate+dlost_thres-1):
                                            for i in range(idf, idf-dlost_thres, -1):
                                                try:
                                                    the_score = frameList[i][[row[0] for row in frameList[i]].index(idx)][5]
                                                except ValueError:
                                                    the_score = -1
                                                if(the_score == 0):
                                                    counter = counter + 1
                                        if(counter < dlost_thres-1):
                                            #detection lost
                                            tracker_bug=0
                                            try:
                                                frameList[idf] = frameList[idf] + [[idx,
                                                                                    frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][1], 
                                                                                    frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][2], 
                                                                                    boundingbox, 
                                                                                    boundingbox,
                                                                                    0]]
                                            except ValueError:
                                                tracker_bug=1
                                            if(tracker_bug==1):
                                                frameList[idf] = frameList[idf] + [[idx,
                                                                                    np.squeeze(classes, axis=0)[max_iou_id], 
                                                                                    np.squeeze(scores, axis=0)[max_iou_id], 
                                                                                    boxes_t[max_iou_id], 
                                                                                    boxes_t[max_iou_id],
                                                                                    0]]
                                        else:
                                            tracks[idx] = []
                                            tracked_boxes[idx] = []
                                else:
                                    #no detection result
                                    frameList[idf] = frameList[idf] + [[idx, 
                                                                        frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][1], 
                                                                        frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][2], 
                                                                        boundingbox, 
                                                                        boundingbox,
                                                                        frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][5]]]

                    #merge
                    m_to_del = []
                    for box1 in frameList[idf]:
                        for box2 in frameList[idf]:
                            if(box1[0] is not box2[0]):
                                iou_m = get_iou(box1[3], box2[3])
                                if(iou_m >= merge_iou):
                                    #delete
                                    tracks[max([box1[0], box2[0]])] = []
                                    tracked_boxes[max([box1[0], box2[0]])] = []
                                    m_to_del = m_to_del + [max([box1[0], box2[0]])]
            
                    new_m = []
                    for i in m_to_del:
                        if i not in new_m:
                            new_m.append(i)
                    for m in new_m:
                        frameList[idf].pop([row[0] for row in frameList[idf]].index(m))               
                            
                    if(video):
                        #visualize detection        
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            write_on,
                            np.array(rev_box_trans([row[3] for row in frameList[idf]])),
                            np.array([row[1] for row in frameList[idf]]).astype(np.int32),
                            np.array([row[2] for row in frameList[idf]]),
                            category_index,
                            use_normalized_coordinates=False,
                            max_boxes_to_draw=500,
                            line_thickness=1,
                            min_score_thresh=0)
            
                        for i in frameList[idf]:
                            cv2.putText(write_on, 
                                (str)(i[0]),
                                ((int)(i[3][0]+i[3][2]/2), (int)(i[3][1]+i[3][3]/2)), 
                                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,51), 2)
                        buffer += [write_on]
     
        end = time.time()
        print('Runs on ', len(frames)/(end - start), ' fps')
        if(video):
            writer = np.array(buffer)
            skvideo.io.vwrite('./output/'+file_name+'_output.mp4', writer)
            #CODE TO WRITE VIDEO WITH BETTER QUALITY
            #to_write = np.array(buffer)
            #writer = skvideo.io.FFmpegWriter(os.path.join(PATH_TO_VIDEO_OUTPUT,PATH_TO_VIDEO.split(sep='/')[-1].split(sep='.')[0]+'_analyzed.avi'), 
            #                         outputdict={'-vcodec': 'libx264', '-b': '2000000', '-framerate': str(fps)})
            #for i in range(to_write.shape[0]):
            #    writer.writeFrame(to_write[i])
            #writer.close()
        export_json(frameList, width, height, fps, file_name, timestamp)

