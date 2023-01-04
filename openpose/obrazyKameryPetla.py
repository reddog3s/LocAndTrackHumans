import cv2
import os
import sys
import re

#potrzebne do importu
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../../python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
import pyopenpose as op

#extract int numbers from string
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

#parameters of RICH sequence
SET = 'train'
SEQ_NAME = 'ParkingLot1_004_005_greetingchattingeating1'
images_path = 'G:/rich-dataset/images' # path to folder with images
camera_ids = list(range(8)) #camera ids to be iterated 

for camera_id in camera_ids:
    flag_interrupt = True
    #parameters passed to openpose (more parameters in include/openpose/flags.hpp)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["net_resolution"] = "-1x368" 
    params["output_resolution"] = "600x400" #resolution of image to display
    #params["write_json"] = os.path.join('C:/Users/PC/inzynierka/videos/pred/rich-dataset/', SET, SEQ_NAME, f'cam_{camera_id:02d}') #write results to given path

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    path = os.path.join(images_path, SET, SEQ_NAME, f'cam_{camera_id:02d}')
    predict_folder = os.listdir(path)

    predict_folder.sort(key=num_sort) #sort files in folder by frame number



    # Process Images in folder
    for file in predict_folder:  
        img_path = os.path.join(path, file)
        datum = op.Datum()
        imageToProcess = cv2.imread(img_path)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Display Image
        #print(file)
        cv2.imshow("OpenPose output", datum.cvOutputData)
        if cv2.waitKey(1) & 0xFF == ord('q'): #stop if q is pressed
            print("Process interrupted for camera %d" % camera_id)
            flag_interrupt = False
            break
    
    if flag_interrupt:
        print("Process done for camera %d!" % camera_id)

#necessary to avoid crash
cv2.destroyAllWindows()