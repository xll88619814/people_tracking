#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from timeit import time
import cv2
cv_version = cv2.__version__
if cv_version.split('.')[0] == '3':
    version = 3
else:
    version = 2
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

def main():
    #output = sys.stdout
    #outputfile = open('log.txt', 'w')
    #sys.stdout = outputfile

    yolo = YOLO()
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True    
    
    video_capture = cv2.VideoCapture('demo.avi')
    #video_capture.set(5, 5)


    if writeVideo_flag:
     # Define the codec and create VideoWriter object
        #w = int(video_capture.get(3))
        #h = int(video_capture.get(4))
        # if version == 3:
        #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # else:
        #     fourcc = cv2.cv.CV_FOURCC(*'MJPG')
        #out = cv2.VideoWriter('output.avi', fourcc, 30, (w, h))
        list_file = open('detection.txt', 'w')
        track_file = open('tracker.txt', 'w')
    
    #print('start', psutil.virtual_memory().available/(1024*1024))
    frame_index = -1
    fps = 0.0
    while True:
        frame_index = frame_index + 1
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        #if (frame_index+1) % 3 != 0:
        #    continue
        t1 = time.time()
        image = Image.fromarray(frame)

        boxs, other_boxs, other_class = yolo.detect_image(image)
        features = encoder(frame, boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        print(frame_index)
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        #print('draw track box', psutil.virtual_memory().available/(1024*1024))
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        for box, label in zip(other_boxs, other_class):
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]+box[0]), int(box[3]+box[1])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1])), 0, 5e-3 * 200, (0, 255, 0), 2)


        if writeVideo_flag:
            # save a frame
            #out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

            track_file.write(str(frame_index)+' ')
            if len(tracker.tracks) != 0:
               for track in tracker.tracks:
                   #print(track.time_since_update, track.age, track.state, track.track_id)
                   bbox = track.to_tlbr()
                   track_file.write(str(track.track_id)+' '+str(track.is_confirmed())+' '+str(track.time_since_update)+' '+str(int(bbox[0])) + ' '+str(int(bbox[1])) + ' '+str(int(bbox[2])) + ' '+str(int(bbox[3])) + ' ')
            track_file.write('\n')
        #print('detect:', time.time()-dt)
        fps = (fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %f"%(fps))


        cv2.imshow('detect result', frame)
        # Press Q to stop!
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        #if frame_index == 10:
        #    break
 

    # print('save .avi')
    yolo.close_session()
    video_capture.release()

    if writeVideo_flag:
        #out.release()
        list_file.close()
        track_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

