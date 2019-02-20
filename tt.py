import cv2
print(cv2.__version__)
from threading import Thread
from people import Tracking
import copy
import time
frames = []

def readframes_(caps):
    global frames
    ind = 0
    while True:
        t1 = time.time()
	cam_no = ind%5
	ind += 1
	for i, cap in enumerate(caps):
	    cap.grab()
        t2 = time.time()
        print('grab time is: ', t2-t1)

	if cam_no < len(caps):
	    ret, frame = caps[cam_no].retrieve()
            frames.append((frame, cam_no))
            print("frames len: {}, cap camera {}, time: {}".format(len(frames), cam_no, time.time()))
            t3 = time.time()
            print('read time is:', t3-t2)
        else:
            time.sleep(0.03)
        


def readframes(caps):
    global frames
    i = -1
    while True:
        i += 1
        for ind, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                if i % 10 == 0:
                    frames.append((frame, ind))

                    print("frames len: {} cap is {} time is {}".format(len(frames), ind, time.time()))
            else:
                print('cap {} ret is False'.format(ind))


def main():
    urls = [#"rtsp://admin:wsy001@192.168.10.8:554/cam/realmonitor?channel=1&subtype=0 ",
            "rtsp://admin:wsy001@192.168.10.8:554/cam/realmonitor?channel=2&subtype=0 ",
            #"rtsp://admin:wsy001@192.168.10.8:554/cam/realmonitor?channel=3&subtype=0 ",
            #"rtsp://admin:wsy001@192.168.10.8:554/cam/realmonitor?channel=4&subtype=0 "
            ]

    caps = []
    for url in urls:
        caps.append(cv2.VideoCapture(url))

    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print("camera not opened")
    
    featmodel = 'model_data/mars-small128.pb'
    tracking = Tracking(featmodel, len(urls))
    im = cv2.imread('first.jpg')
    for i in range(len(urls)):
        result = tracking.tracking_people(im, i)

    thread = Thread(target=readframes_, args=(caps, ))
    thread.setDaemon(True)
    thread.start()

    while True:
        if frames != []:
            starttime = time.time()
            frame = frames[0]
            print('process:', frame[1])
            result = tracking.tracking_people(frame[0], frame[1])
            #result = frame[0]
            windowname = str(frame[1])
            cv2.imshow(windowname, result)
            frames.remove(frames[0])
            endtime = time.time()
            print('tracking time is: ', endtime - starttime)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    main()

