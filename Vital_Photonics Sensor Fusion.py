import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from ultralytics import YOLO
import math
import datetime as dt
import matplotlib.animation as animation
import threading
from enum import Enum
import queue
import time
import pytesseract
from mss import mss
from PIL import Image

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

outRGB = cv2.VideoWriter('outRGB.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
outLWIR =cv2.VideoWriter('outLWIR.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
record = False

# global variable for tracking position
prev_center = [0, 0]
cent_x = [0, 0, 0, 0]
cent_y = [0, 0, 0, 0]
prev_cent_x = [0, 0, 0, 0]
prev_cent_y = [0, 0, 0, 0]
xdiff = [0, 0, 0, 0]
ydiff = [0, 0, 0, 0]

# Global 
rgbData = (0, 0)
lwirData = (0, 0)
largeMove = 0
smallMove = 0

# define constants
HF_MVMT_THRESH = 1.4
LF_MVMT_THRESH = 0.1

class CameraType(Enum):
    RGB = 1
    THERMAL = 2

class thermal:
    def raw_to_8bit(data):
        cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(data, 8, data)
        return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

    def ktoc(val):
        return (val - 27315) / 100.0
    
    def display_temperature(img, val_k, loc, color):
        val = thermal.ktoc(val_k)
        cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        x, y = loc
        cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
        cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

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
        if self.cameraType == CameraType.THERMAL:
            cap = cv2.VideoCapture(camID)
        else:
            cap = cv2.VideoCapture(camID,cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Camera not found!")
            exit(1)
        self.cap = cap
    def run(self):
        print("Starting " + self.previewName)
        camView(self)

def camView(self):
    cv2.namedWindow(self.previewName, cv2.WINDOW_NORMAL)

    if self.cap.isOpened():  # try to get the first frame
        width  = self.cap.get(3)  # float `width`
        height = self.cap.get(4)  # float `height`
        self.cap.set(cv2.CAP_PROP_FPS,15)
        fps = self.cap.get(5)
        cutoff = 3
        RC = 1/(cutoff * 2 * 3.141)
        dt = 1/fps
        alpha = dt/(RC + dt)
        gen_move = 0
        filt_gen_move = 0
        past_gen_move = 0
        ret, frame = self.cap.read()
    else:
        ret = False

    initTime = time.time() * 1000
    global rgbData
    global lwirData
    
    if self.cameraType == CameraType.RGB :
        global smallMove
        global largeMove
        cX = [0,0,0,0]
        cY = [0,0,0,0]

        while ret:
            # capture from the camera
            ret, frame = self.cap.read()
            results = self.model.predict(frame, stream=True, classes=0, conf = 0.5, verbose=False)
            capTime = time.time() * 1000

            # coordinates
            for r in results:
                boxes = r.boxes
                i = 0

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # confidence
                    # confidence = math.ceil((box.conf[0]*100))/100
                    # print("Confidence --->",confidence)

                    # find center of box. Scale so that the center is 0,0 
                    cX[i] = (x2 - (x2 - x1)/2)
                    cent_x[i] = cX[i] - width/2
                    cY[i] = (y2 - (y2 - y1)/2)
                    cent_y[i] = cY[i] - height/2
                    cv2.circle(frame, (int(cX[i]), int(cY[i])), 2, (255, 255, 255), -1)

                    xdiff[i] = cent_x[i]-prev_cent_x[i]
                    ydiff[i] = cent_y[i]-prev_cent_y[i]
                    gen_move = (abs(xdiff[i]) + abs(ydiff[i]))*2
                    filt_gen_move = past_gen_move + (alpha * (gen_move - past_gen_move))
                    past_gen_move = filt_gen_move

                    rgbMutex.acquire()
                    if filt_gen_move > 50:
                        largeMove += filt_gen_move
                        # put box in cam
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    elif filt_gen_move > 6:
                        smallMove += filt_gen_move
                        # put box in cam
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    else:
                        # put box in cam
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    rgbMutex.release()

                    
                
                    vol = ((x2-x1)*(y2-y1)) * 0.001

                    # prev_direction = direction
                    prev_cent_x[i] = cent_x[i]
                    prev_cent_y[i] = cent_y[i]

                    # index for the number of results
                    i = i + 1

            # wait for the q button to be pressed to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                outRGB.release()
                break

            # plot1 = int(cent_x[0])
            # plot2 = int(cent_y[0])
            # p.multiplot([int(gen_move), int(filt_gen_move)])

            if(capTime - initTime >= 66.67 ):
                rgbMutex.acquire()
                rgbData = (cent_x[0], cent_y[0], smallMove)
                rgbMutex.release()
                initTime = round(time.time() * 1000)

            cv2.imshow(self.previewName, frame)
            if(record):
                outRGB.write(frame)
                
        self.cap.release()
        cv2.destroyWindow(self.previewName)

    elif self.cameraType == CameraType.THERMAL :

        while ret:
            # capture from the camera
            ret, frame = self.cap.read()
            gray1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            capTime = time.time() * 1000

            ret1, thresh = cv2.threshold(gray1, 120,255, cv2.THRESH_BINARY)

            # calculate moments of binary image
            M = cv2.moments(thresh)
            
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])   
            cY = int(M["m01"] / M["m00"])
            centroid = (cX - width/2, cY - height/2)
            # put text and highlight the center
            cv2.circle(frame, (cX, cY), 2, (255, 255, 255), -1)
            cv2.putText(frame, "Detected", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if(capTime - initTime >= 66.67 ):
                lwirMutex.acquire()
                lwirData = centroid
                lwirMutex.release()
                initTime = time.time() * 1000

            # wait for the q button to be pressed to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                outLWIR.release()
                break


            cv2.imshow(self.previewName, cv2.resize(frame, (640,480)))
            if (record):
                outLWIR.write(cv2.resize(frame, (640,480)))
            # gray2 = gray1
                
        self.cap.release()
        cv2.destroyWindow(self.previewName)
    else: print("Camera type not recognised \n")

class fusionThread(threading.Thread):
    def __init__(self,rateQueue):
        threading.Thread.__init__(self)
        self.rateQueue = rateQueue
    def run(self):
        print("Fusion thread starting")

        # calibration constants for aligning the thermal and RGB cameras
        lwirXOffset = 0           
        lwirYOffset = 0
        lwirXScaler = 4           
        lwirYScaler = 4
        global rgbData
        global lwirData
        global largeMove
        global smallMove

        dataOut = []

        while True:

            # Block until there is a rate on the queue in which case get it to process
            rate = rateQueue.get(block=True)
            rateQueue.task_done()

            if (isinstance(rate, int)):
                # Get the RGB and LWIR centroid data 
                # check if person is detected on RGB Camera by seeing if anything is on the rgb queue
                # note currently this only allows for a single person to be detected
                rgbMutex.acquire()
                rgbVal = (rgbData[0], rgbData[1], smallMove, largeMove) #queue values are tuples of (x, y, LF, HF) i.e. centroid x value, centroid y value, low-frequency movement coefficeint, high frequency movement coefficient
                smallMove = largeMove = 0
                rgbMutex.release()
                
                # check if anything is detected on the IR camera
                lwirMutex.acquire()
                lwirVal = lwirData
                lwirMutex.release()
            
                # extract, scale and check that the centroid values line up
                rgbCentroid = (rgbVal[0], rgbVal[1])
                lwirCentroid = rgbCentroid#((lwirVal[0] + lwirXOffset) * lwirXScaler, (lwirVal[1] + lwirYOffset) * lwirYScaler)
                centroidDiff = tuple(np.abs(np.subtract(np.abs(rgbCentroid),np.abs(lwirCentroid))))

                # check if the RGB and LWIR positionings match - get the current calculated respiratory rate
                centroidThresh = 40 
                if ((centroidDiff[0] < centroidThresh) & (centroidDiff[1] < centroidThresh)):
                    if (rgbVal[3] > 0):
                        dataOut.append((rate,1))
                        print("resp rate detected but issue with large movements")
                    elif (rgbVal[2] > 40):
                        print("resp rate detected but issue with small movements")
                        dataOut.append((rate,2))
                    else:
                        print("respiratory rate detected")
                        # if (rate < 10):
                        #     dataOut.append((rate,0))
                        #     print("ALARM - LOW RESP RATE - " + str(rate) + "bpm")
                        # elif (rate > 25):
                        #     dataOut.append((rate,0))
                        #     print("ALARM - HIGH RESP RATE - " + str(rate) + "bpm")
                        # else:
                        #     dataOut.append((rate,0))
                        #     print("respiratory rate = " + str(rate))
                else:
                    print("resp rate calculated but issue with thermal alignment")
            elif  (rate == 'z'):
                print("resp rate detected but issue with small movements")
                
            else:
                print("resp rate detected but issue with small movements")
                

            #     # check the movement coefficients of the target and publish the respiratory rate 
            #     if (rgbVal[2] < LF_MVMT_THRESH) & (rgbVal[3] < HF_MVMT_THRESH):
            #         print("measured resp rate = " + rate + '\n')
            #     elif (rgbVal[2] > LF_MVMT_THRESH):
            #         print("Low Frequency movement above threshold\n")
            #     else:
            #         print("HIGH Frequency movement above threshold\n")


class eRadarThread(threading.Thread):
    def __init__(self, respData, avgSize, rateQueue):
        threading.Thread.__init__(self)
        self.respData = respData
        self.avgSize = avgSize
        self.rateQueue = rateQueue

    def run(self):

        length = len(self.respData)

        for x in range(length):
            # Replicating a live stream of a single resp rate value coming from the hardware
            # Get the data from the array
            currentRate = self.respData[x]

            # Set the value of the global variable
            rateQueue.put(currentRate)
            if(currentRate == 'z'):
                currentRate = 24
            time.sleep(60/currentRate)
        print("no more resp rates in data set")
        
                


## Start of Program ###
if __name__ == '__main__':
    # Initialize plot.
    # p = Plotter(400, 480, 2) #(plot_width, plot_height)   
    # Create queue for passing resprate data
    rateQueue = queue.Queue(maxsize=0)

    # Import the sample respiration data
    respData = [22,21,24,21,24,22,24,24,'z','z','z','z',24,22,22,22,'z','z',22,24] # example of data that could be fed out of the eradar - an array of respiratory rates
    rateAvgSize = 3                                     # number of samples used to calculate the respData above
    numRates = len(respData)                            # find the number of resp rates in total

    # Create a mutex for ensuring the resp rate values are not accessed at the same time by the multiple threads
    rgbMutex = threading.Lock()
    lwirMutex = threading.Lock()

    # Create the first thread with the RGB Camera, passing in the YOLO model
    # LWIR_THREAD = camThread("LWIR Camera", 0, CameraType.THERMAL, 1)
    # RGB_THREAD = camThread("RGB Camera", 2, CameraType.RGB, YOLO("weights/yolov8n.pt"))
    FUSION_THREAD = fusionThread(rateQueue)
    ERADAR_THREAD = eRadarThread(respData, rateAvgSize, rateQueue)

    # LWIR_THREAD.start()
    # RGB_THREAD.start()

    time.sleep(10)
    FUSION_THREAD.start()
    ERADAR_THREAD.start()


""" Code for using OCR to extract rate from PCR radar
    bounding_box = {'top': 980, 'left': 250, 'width': 150, 'height': 130}
        sct = mss()
        
        while True:
            sct_img = np.array(sct.grab(bounding_box))
        
            # convert to grayscale
            gray = cv2.cvtColor(sct_img, cv2.COLOR_RGB2GRAY)

            # Performing OTSU threshold
            ret, thresh1 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

            # Apply OCR 
            text = pytesseract.image_to_string(thresh1, config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.')
            if not text:
                rateQueue.put('z')
            else:
                try:
                    rate = float((text.strip()))     
                except:
                    rateQueue.put('z')
                    break

                if (rate > 40):
                    rateQueue.put('x')
                elif(rate < 5):
                    rateQueue.put('x')
                else:
                    rateQueue.put(rate)

            time.sleep(3) """