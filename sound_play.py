#! /usr/bin/env python

import subprocess
import threading
import time

play_flag = False
audio_file = "mp3/warning.mp3"

class myThread(threading.Thread):
    def __init__(self, func):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.func = func
    def run(self):
        while True:
            self.func()
            time.sleep(0.1)
        
def play():
    global play_flag, audio_file
    if play_flag:
        subprocess.call("play {} > /dev/null 2>&1".format(audio_file), shell=True)
        play_flag = False


def sound_play(file_name = "mp3/warning.mp3"):
    global play_flag, audio_file
    audio_file = file_name
    play_flag = True
    # play("mp3/warning.mp3", stuck=False)


t = myThread(play)
t.start()


if __name__ == "__main__":
    play_flag=True
    sound_play()
    while True:
        print(play_flag)
        time.sleep(1)
