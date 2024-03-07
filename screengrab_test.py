import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
import pytesseract

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

bounding_box = {'top': 980, 'left': 250, 'width': 150, 'height': 130}
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

sct = mss()

while True:
    ret, frame = cap.read()
    cv2.imshow('test', frame)
    # screen grab
    sct_img = np.array(sct.grab(bounding_box))
    
    # convert to grayscale
    gray = cv2.cvtColor(sct_img, cv2.COLOR_RGB2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    # Apply OCR 
    text = pytesseract.image_to_string(thresh1, config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.')

    print(text)

    cv2.imshow('screen', thresh1)


    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

    time.sleep(1)

    