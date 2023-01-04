import cv2 as cv
import os
import sys
import time
import numpy as np

#necessary to import openpose library
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../../python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
import pyopenpose as op

#parameters passed to openpose (more parameters in include/openpose/flags.hpp)
params = dict()
params["model_folder"] = "../../../models/"
params["net_resolution"] = "368x160" 
params["output_resolution"] = "600x400" #resolution of image to display

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()



# Display and Write Video

cap = cv.VideoCapture("C:/Users/Dell/inzynierka/videos/Bieg1.mp4")
size = (cap.get(3), cap.get(4)) #input resolution of video


result = cv.VideoWriter('C:/Users/PC/inzynierka/output.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                        15, size) #write video in avi format, 15 fps, resolution the same as input

if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")

times = []

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    # Process Image
    st = time.time()
    datum = op.Datum()
    imageToProcess = frame
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    et = time.time()

    elapsed_time = et - st
    times.append(elapsed_time)

    cv.imshow('OpenPose output', datum.cvOutputData)
    
    result.write(datum.cvOutputData)

times = np.array(times)
print("Average time: ", np.average(times)) #average time of processing frames [s/img]

#necessary to avoid crash
cap.release()
result.release()
cv.destroyAllWindows()