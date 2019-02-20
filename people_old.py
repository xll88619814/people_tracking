from __future__ import division, print_function, absolute_import
from timeit import time
import cv2
# cv_version = cv2.__version__
# if cv_version.split('.')[0] == '3':
#     version = 3
# else:
#     version = 2
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class Params:
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0


class Tracking:
    def __init__(self, featmodel, cap_num, num):
        self.cap_num = cap_num
        self.yolo = YOLO()
        self.encoder = gdet.create_box_encoder(featmodel, batch_size=1)
        self.time = 1.0/(30.0/num-1)
        self.alltrackers = []
        self.preframe = {}
        self.velo = {}
        for i in range(cap_num):
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", Params.max_cosine_distance, Params.nn_budget)
            self.alltrackers.append(Tracker(metric))
            self.preframe.update({i:[]})
            self.velo.update({i:[]})
        print(self.preframe, self.time)

    def tracking_people(self, frame, index):
        starttime = time.time()

        image = Image.fromarray(frame)

        t1 = time.time()
        boxs, other_boxs, other_class = self.yolo.detect_image(image)
        if index == self.cap_num:
            return []
        t2 = time.time()
        #print('detect time is........:', t2-t1)
        features = self.encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, Params.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        #print(self.alltrackers[index])
        self.alltrackers[index].predict()
        self.alltrackers[index].update(detections)

        track_boxes = []
        velo_list = []
        for track in self.alltrackers[index].tracks:
            if track.is_confirmed() and track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            display = str(track.track_id)
            center_box = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
            track_boxes.append((center_box, track.track_id))

            #if len(self.preframe[index]) > 0:
            while len(self.preframe[index]) > 0:
                if self.preframe[index][0] == [] :
                    del self.preframe[index][0]
                else:
                    #if len(self.preframe[index]) == 0:
                    break
            if len(self.preframe[index]) > 0: 
                for (pre_center_box, pre_track_id) in self.preframe[index][0]:
                    time_diff = self.time * len(self.preframe[index])
                    if track.track_id == pre_track_id:
                        dist = np.sqrt(np.square(abs(pre_center_box[0]-center_box[0]))+np.square(abs(pre_center_box[1]-center_box[1])))
                        velocity_new = round(dist*0.004/time_diff, 2)
                        for (velocity_old, velo_id) in self.velo[index]:
                            if velo_id == pre_track_id:
                                velocity_new = round((velocity_new+velocity_old)/2, 2)
                                break
                        velo_list.append((velocity_new, track.track_id))
                        if velocity_new > 0.3:
                            display = display+' '+str(velocity_new)+' walking'
                        else:
                            display = display+' '+str(velocity_new)+' stopping'
                        break 
                        
            cv2.putText(frame, display, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
            
        if len(self.preframe[index]) == 1:
            del self.preframe[index][0]
            self.preframe[index].append(track_boxes)
        else:
            self.preframe[index].append(track_boxes)
        self.velo.update({index:velo_list})

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        endtime = time.time()
        #print('tracking time is: ', endtime - starttime)

        return frame



