import cv2
import os
import sys
import re
import time
import numpy as np

#necessary to import openpose library
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../../python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
import pyopenpose as op

#extract int numbers from string
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

#parameters of RICH sequence
SET = 'train'
SEQ_NAME = 'Pavallion_003_018_tossball'
CAMERA_ID = 5
images_path = 'G:/rich-dataset/images' # path to folder with images

#parameters passed to openpose (more parameters in include/openpose/flags.hpp)
params = dict()
params["model_folder"] = "../../../models/"
params["net_resolution"] = "-1x368" 
params["output_resolution"] = "600x400" #resolution of image to display
#params["write_json"] = os.path.join('C:/Users/PC/inzynierka/videos/pred/rich-dataset/', SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}') #write results to given path

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


path = os.path.join(images_path, SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}')
predict_folder = os.listdir(path)

predict_folder.sort(key=num_sort) #sort files in folder by frame number

times = []

flag_interrupt = True

# Process Images in folder
for file in predict_folder:  
    img_path = os.path.join(path, file)
    imageToProcess = cv2.imread(img_path)

    st = time.time()
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    et = time.time()
    
    elapsed_time = et - st
    times.append(elapsed_time)

    # Display Image
    print(file)
    cv2.imshow("OpenPose output", datum.cvOutputData)
    if cv2.waitKey(1) & 0xFF == ord('q'): #stop if q is pressed
        print("Process interrupted")
        flag_interrupt = False
        break

if flag_interrupt:
    print("Process done!")

times = np.array(times)
print("Average time: ", np.average(times)) #average time of processing frames [s/img]
#necessary to avoid crash
cv2.destroyAllWindows()