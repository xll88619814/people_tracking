from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

if __name__ == '__main__':
    cap = cv2.VideoCapture('output.avi')
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite('images/'+str(i)+'.jpg', frame)
        i += 1
    cap.release()