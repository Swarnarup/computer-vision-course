import cv2
import numpy as np
import matplotlib.pyplot as plt

vid = cv2.VideoCapture("video.mp4")

a,b = vid.read()

k = 24

a=True
cnt=1
while a:
    a,b=vid.read()
    if cnt%k == 0:
        res_b = cv2.resize(b,(256,256))
        filename = "frame "+str(cnt)+".jpg"
        cv2.imwrite("video/"+filename,res_b)
    
    cnt += 1