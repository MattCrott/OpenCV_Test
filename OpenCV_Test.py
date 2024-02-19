import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from ultralytics import YOLO
import math
import datetime as dt
import matplotlib.animation as animation
import threading
from enum import Enum
from queue import Queue

# global variable for tracking position
prev_center = [0, 0]
cent_x = [0, 0, 0]
cent_y = [0, 0, 0]
prev_cent_x = [0, 0, 0]
prev_cent_y = [0, 0, 0]
xdiff = [0, 0, 0]
ydiff = [0, 0, 0]

# global variables
respRate = 0

# define constants
HF_MVMT_THRESH = 1.4
LF_MVMT_THRESH = 0.1

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
    elif cameraType == CameraType.THERMAL :
        while ret:
            # capture from the camera
            ret, frame = cap.read()
            blobResults = model.detect(frame)

            frame = cv2.drawKeypoints(frame, blobResults, np.arrayp[])
        

            # wait for the q button to be pressed to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


            cv2.imshow(previewName, frame)
                
        cap.release()
        cv2.destroyWindow(previewName)
    else: print("Camera type not recognised \n")


class fusionThread(threading.Thread):
    def __init__(self, rgbQueue, lwirQueue):
        threading.Thread.__init__(self)
        self.rgbQueue = rgbQueue
        self.lwirQueue = lwirQueue
    def run(self):
        # calibration constants for aligning the thermal and RGB cameras
        lwirXOffset = 10            # TODO find values to offset the LWIR camera - will be dependent on final casing design
        lwirYOffset = 10
        lwirXScaler = 1.2           # TODO find values to scale X and Y centroid for lwir onto rgb
        lwirYScaler = 1.1

        while True:

            # check if person is detected on RGB Camera by seeing if anything is on the rgb queue
            # note currently this only allows for a single person to be detected
            while not rgbQueue.empty():
                rgbVal = rgbQueue.get() #queue values are tuples of (x, y, LF, HF) i.e. centroid x value, centroid y value, low-frequency movement coefficeint, high frequency movement coefficient
                rgbQueue.task_done()
            
            # check if anything is detected on the IR camera
            while not lwirQueue.empty():
                lwirVal = lwirQueue.get()
                lwirQueue.task_done()
            
            # if a person and a heat blob has been detected
            if (('rgbVal' in locals()) & ('lwirVal' in locals())):

                # extract, scale and check that the centroid values line up
                rgbCentroid = (rgbVal[0], rgbVal[1])
                lwirCentroid = ((lwirVal[0] + lwirXOffset) * lwirXScaler, (lwirCentroid[1] + lwirYOffset) * lwirYScaler)
                if(abs(rgbCentroid - lwirCentroid) < 20):
                    # the RGB and LWIR positionings match - get the current calculated respiratory rate
                    print("thermal and rgb target aligned")
                    respMutex.acquire()
                    rate = respRate
                    respMutex.release()

                    # check the movement coefficients of the target and publish the respiratory rate 
                    if (rgbVal[2] < LF_MVMT_THRESH) & (rgbVal[3] < HF_MVMT_THRESH):
                        print("measured resp rate = " + rate + '\n')
                    elif (rgbVal[2] > LF_MVMT_THRESH):
                        print("Low Frequency movement above threshold\n")
                    else:
                        print("HIGH Frequency movement above threshold\n")

                else: 
                    print("thermal and rgb target not aligned")


            del rgbVal
            del lwirVal 


class eRadarThread(threading.Thread):
    def __init__(self, respData, avgSize):
        threading.Thread.__init__(self)
        self.respData = respData
        self.avgSize = avgSize
    def run(self):
        while True:
            # Replicating a live stream of a single resp rate value coming from the hardware
            # Get the data from the array
            rate = respData(1)
            # Wait the required amount of time 

            # Set the value of
            respMutex.acquire()
            respRate = rate
            respMutex.release()





## Start of Program ###

# Initialize plot.
p = Plotter(400, 480, 2) #(plot_width, plot_height)   

# Create two queues for passing 
rgbQueue = Queue(maxsize=0)
lwirQueue = Queue(maxsize=0)

# Import the sample respiration data
respData = [14, 14, 14, 14, 14, 15, 14, 16, 15, 14] # example of data that could be fed out of the eradar - an array of respiratory rates
rateAvgSize = 3                                     # number of samples used to calculate the respData above
numRates = len(respData)                            # find the number of resp rates in total

# Create a mutex for ensuring the resp rate values are not accessed at the same time by the multiple threads
respMutex = threading.Lock()


# Create the first thread with the RGB Camera, passing in the YOLO model
thread1 = camThread("Camera 1", 0, CameraType.RGB, YOLO("yolo0-Weights/yolov8n.pt"))
# thread2 = camThread("Camera 2", 2)
thread3 = fusionThread(,)
thread1.start()
# thread2.start()



    