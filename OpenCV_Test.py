import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from ultralytics import YOLO
import math


# manual tracker selection setup tracker
# tracker = cv2.TrackerCSRT_create()
# BB = None
# def track(frame):
#     (success, box) = tracker.update(frame)
#     if success:
#         (x,y,w,h) = [int(v) for v in box]
#         cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
#     return success, frame

# setup HOG
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# setup YOLO
model = YOLO("yolo0-Weights/yolov8n.pt")

# Object detection, manually selecting the object using tracker 
# while cap.isOpened():
#     # capture from the camera
#     ret, frame = cap.read()

#     #check bounding box
#     if BB is not None:
#         success, frame = track(frame)

#     cv2.imshow('Webcam', frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('q'):
#         break
#     elif key == ord('c'):
#         BB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
#         tracker.init(frame, BB)
# cap.release()
# cv2.destroyAllWindows()

# Pedestrian detection using HOG
# while cap.isOpened():

#     # capture from the camera
#     ret, frame = cap.read()

#     #detect people in the image
#     boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))

#     boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

#     for (xA, yA, xB, yB) in boxes:
#         cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

#     cv2.imshow('Webcam', frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#object detection using YOLO
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # capture from the camera
    ret, frame = cap.read()
    results = model.predict(frame, stream=True, classes=0, conf = 0.7, verbose=False)
        
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->",confidence)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(frame, "person", org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()