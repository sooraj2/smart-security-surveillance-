import cv2
import numpy as np
import time
import os
from datetime import datetime
import smtplib
#from twilio.rest import Client  

# the following line needs your Twilio Account SID and Auth Token
#Uncomment below code if you want to use message feature and insert your credentials

#client = Client("Account SID", "Auth Token")


def send_email(subject, msg):
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login('soorajkumar974@gmail.com', 'waheguru!@#7') #insert you email and password
        message = 'Subject: {}\n\n{}'.format(subject, msg)
        server.sendmail('soorajkumar974@gamil.com','hafeezmaitlo1992@gmail.com', message) #safdarsoomro@iba-suk.edu.pk insert your email and recievers email
        server.quit()
        print("Success: Email sent!")
    except:
        print("Email failed to send.")


subject = "Threat has been Detected."
msg = "You are informed that a threat has been found in your X room, kindly proceed as fast as you can. \nThis is computer generated mail sent from using python. A part of AI project. \nregards, \nSooraj Kumar."


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




# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Loading camera
cap = cv2.VideoCapture(0)
vr = cv2.VideoWriter(getfilename(), get_video_type(filename), 25, get_dims(cap, res))  

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape
        # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:] #first 4 elements center_x, center_y, width and height
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3) #non maximim supression

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=="person" or label=="knife":
                vr.write(frame)
                if label=="knife":
                    print("threat is here...")
                    if(check):
                        send_email(subject, msg)
                        #Message code costs $0.06 per message 
                        #uncomment below code if you wnat to use message notification and insert your twilio's given number and and recievers number
                        '''client.messages.create(to="recievers number", 
                       from_="your twilio account number", 
                       body="The threat has been found by intelligent security system.\n regards, \n sooraj.")'''
                        check=False
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
vr.release()
cap.release()
cv2.destroyAllWindows()