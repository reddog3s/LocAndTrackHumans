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
params["net_resolution"] = "-1x368"
params["output_resolution"] = "600x400" #resolution of image to display
#params["write_json"] = 'C:/Users/PC/inzynierka/videos/pred/dron/bieg2/' #write results to given path

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()



# Display Image

cap = cv.VideoCapture('C:/Users/Dell/inzynierka/dron/Bieg1.MP4') #open video cap

if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")

times = [] #times of processing frames by openpose

while cv.waitKey(1) < 0:
    hasFrame, imageToProcess = cap.read()
    if not hasFrame:
        cv.waitKey() #stop if there isnt any frame
        break
    # Process Image
    st = time.time()
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    et = time.time()

    elapsed_time = et - st
    times.append(elapsed_time)

    cv.imshow('Pose estimation', datum.cvOutputData)


print("Process done!")
times = np.array(times)
print("Average time: ", np.average(times)) #average time of processing frames [s/img]

#necessary to avoid crash
cap.release()
cv.destroyAllWindows()