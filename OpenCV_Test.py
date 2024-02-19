import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from ultralytics import YOLO
import math
import datetime as dt
import matplotlib.animation as animation
import threading
from enum import Enum

# global variable for tracking position
prev_center = [0, 0]
cent_x = [0, 0, 0]
cent_y = [0, 0, 0]
prev_cent_x = [0, 0, 0]
prev_cent_y = [0, 0, 0]
xdiff = [0, 0, 0]
ydiff = [0, 0, 0]

class CameraType(Enum):
    RGB = 1
    THERMAL = 2

# Create class used to graph data in an opencv image
class Plotter:
    def __init__(self, plot_width, plot_height, num_plot_values):
        self.width = plot_width
        self.height = plot_height
        self.color_list = [(255, 0 ,0), (0, 250 ,0),(0, 0 ,250),
                    (0, 255 ,250),(250, 0 ,250),(250, 250 ,0),
                    (200, 100 ,200),(100, 200 ,200),(200, 200 ,100)]
        self.color  = []
        self.val = []
        self.plot = np.ones((self.height, self.width, 3))*255

        for i in range(num_plot_values):
            self.color.append(self.color_list[i])

    # Update new values in plot
    def multiplot(self, val, label = "plot"):
        self.val.append(val)
        while len(self.val) > self.width:
            self.val.pop(0)

        self.show_plot(label)

    # Show plot using opencv imshow
    def show_plot(self, label):
        self.plot = np.ones((self.height, self.width, 3))*255
        cv2.line(self.plot, (0, int(self.height/2) ), (self.width, int(self.height/2)), (0,255,0), 1)
        for i in range(len(self.val)-1):
            for j in range(len(self.val[0])):
                cv2.line(self.plot, (i, int(self.height/2) - self.val[i][j]), (i+1, int(self.height/2) - self.val[i+1][j]), self.color[j], 1)

        cv2.putText(self.plot,"x",[10,500],cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(self.plot,"y",[10,525],cv2.FONT_HERSHEY_SIMPLEX,1,(0,250,0),2)
        # cv2.putText(self.plot,"vol",[10,550],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),2)
        cv2.imshow(label, self.plot)
        cv2.waitKey(10)

class camThread(threading.Thread):
    def __init__(self, previewName, camID, cameraType, model):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.cameraType = cameraType
        self.model = model
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID, self.cameraType, self.model)

def camPreview(previewName, camID, cameraType, model):
    cv2.namedWindow(previewName)
    cap = cv2.VideoCapture(camID,cv2.CAP_DSHOW)

    if cap.isOpened():  # try to get the first frame
        width  = cap.get(3)  # float `width`
        height = cap.get(4)  # float `height`
        cap.set(cv2.CAP_PROP_FPS,15)
        fps = cap.get(5)
        cutoff = 3
        RC = 1/(cutoff * 2 * 3.141)
        dt = 1/fps
        alpha = dt/(RC + dt)
        gen_move = 0
        filt_gen_move = 0
        past_gen_move = 0
        ret, frame = cap.read()
        cv2.imshow(previewName, frame)
    else:
        ret = False

    if cameraType == CameraType.RGB :
        while ret:
            # capture from the camera
            ret, frame = cap.read()
            results = model.predict(frame, stream=True, classes=0, conf = 0.7, verbose=False)

            # coordinates
            for r in results:
                boxes = r.boxes
                i = 0

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # put box in cam
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    # print("Confidence --->",confidence)

                    # find center of box. Scale so that the center is 0,0 
                    cent_x[i] = (x2 - (x2 - x1)/2) - width/2
                    cent_y[i] = (y2 - (y2 - y1)/2) - height/2

                    xdiff[i] = cent_x[i]-prev_cent_x[i]
                    ydiff[i] = cent_y[i]-prev_cent_y[i]
                    gen_move = (abs(xdiff[i]) + abs(ydiff[i]))*2
                    filt_gen_move = past_gen_move + (alpha * (gen_move - past_gen_move))
                    past_gen_move = filt_gen_move

                    if filt_gen_move > 40:
                        print("large movement alert")
                    elif filt_gen_move > 10:
                        print("small movement")
                    

                    # object details
                    org = [x1, y1-10]
                    org2 = [x1 + 150, y1-10]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 255)
                    thickness = 1
                
                    vol = ((x2-x1)*(y2-y1)) * 0.001

                    # prev_direction = direction
                    prev_cent_x[i] = cent_x[i]
                    prev_cent_y[i] = cent_y[i]
                    cv2.putText(frame, f"loc {cent_x[i]},{cent_y[i]}", org, font, 0.5, color, thickness)
                    # cv2.putText(frame, direction, org2, font, fontScale, color, 2)
                    i = i + 1

            # wait for the q button to be pressed to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            plot1 = int(cent_x[0])
            plot2 = int(cent_y[0])
            p.multiplot([int(gen_move), int(filt_gen_move)])
            cv2.imshow(previewName, frame)
                
        cap.release()
        cv2.destroyWindow(previewName)

## Start of Program

# setup YOLO
# model1 = YOLO("yolo0-Weights/yolov8n.pt")

# Initialize plot.
p = Plotter(400, 480, 2) #(plot_width, plot_height)   

# Create the first thread with the RGB Camera, passing in the YOLO model
thread1 = camThread("Camera 1", 0, CameraType.RGB, YOLO("yolo0-Weights/yolov8n.pt"))
# thread2 = camThread("Camera 2", 2)
thread1.start()
# thread2.start()



    