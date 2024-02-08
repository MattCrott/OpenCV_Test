import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from ultralytics import YOLO
import math
import datetime as dt
import matplotlib.animation as animation

# global variable for tracking position
prev_center = [0, 0]
cent_x = [0, 0, 0]
cent_y = [0, 0, 0]
prev_cent_x = [0, 0, 0]
prev_cent_y = [0, 0, 0]
xdiff = [0, 0, 0]
ydiff = [0, 0, 0]

# Plot values in opencv program
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

# setup YOLO
model = YOLO("yolo0-Weights/yolov8n.pt")

# Initialize plot.
p = Plotter(400, 480, 2) #(plot_width, plot_height)   

#object detection using YOLO
cap = cv2.VideoCapture(0)
width  = cap.get(3)  # float `width`
height = cap.get(4)  # float `height`
fps = cap.get(5)
cutoff = 3
RC = 1/(cutoff * 2 * 3.141)
dt = 1/fps
alpha = dt/(RC + dt)
past_gen_move = 0

while cap.isOpened():
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
            
            # add direction of movement to
            # scaler = 1
            # if center[0] > prev_center[0] + scaler:
            #     direction = "left"
            # elif center[0] < prev_center[0] - scaler:
            #     direction = "right"
            # if center[1] > prev_center[1] + scaler:
            #     direction = "up"
            # elif center[1] < prev_center[1] - scaler:
            #     direction = "down"
            # else:
                # direction = ""

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
    cv2.imshow('Webcam', frame)
    

    

cap.release()
cv2.destroyAllWindows()