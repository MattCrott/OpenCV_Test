import cv2
import requests
import matplotlib.pyplot as plt
import torch
from skimage import io
from PIL import Image
from SimpleHRNet import SimpleHRNet
import time
from misc.visualization import joints_dict
import numpy as np
import urllib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# multiperson w/ YOLOv3, COCO weights
model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", device=device)

# # multiperson w/ YOLOv3, COCO weights, small model
# model = SimpleHRNet(32, 17, "./weights/pose_hrnet_w32_256x192.pth", device=device)

def plot_joints(frame, output):
        bones = joints_dict()["coco"]["skeleton"]
        # bones = joints_dict()["mpii"]["skeleton"]

        for bone in bones:
            xS = [output[:,bone[0],1], output[:,bone[1],1]]
            yS = [output[:,bone[0],0], output[:,bone[1],0]]
            cv2.line(frame, (round(xS[0][0]), round(yS[0][0])), (round(xS[1][0]), round(yS[1][0])),  (0,255,0),3)
        for j in range(len(joints[0,:,0])):
            cX = joints[0,j,1]
            cY = joints[0,j,0]
            cv2.circle(frame, (int(cX), int(cY)) ,2,(0,0,255), -1)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS,15)
ret, frame = cap.read()
joints = model.predict(frame)
plot_joints(frame,joints)

cv2.imshow('test', frame)
time.sleep(3)

cv2.imwrite('savedImage.jpg', frame)

input("Press Enter to continue...")

# key = cv2.waitKey(1) & 0xFF
# if key == ord('q'):
#     break



