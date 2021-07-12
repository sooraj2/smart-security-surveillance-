import numpy as np
import cv2
import os
from datetime import datetime

filename = 'video1.avi'
frames_per_second = 24.0
res = '720p'
check=True

def getfilename():
	now=datetime.now()
	name=now.strftime("%m_%d_%Y_%H_%M_%S")
	name=name + ".avi"
	return name 

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']




face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')


cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(getfilename(), get_video_type(filename), 25, get_dims(cap, res))  
while(True):
    
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
    	out.write(frame)
    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
	

	
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

	

out.release()
cap.release()
cv2.destroyAllWindows()