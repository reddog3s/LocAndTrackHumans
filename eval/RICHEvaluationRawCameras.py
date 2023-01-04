import os
import json
import pickle

from mojeFunkcjePCh import *
from mojeFunkcjeAP import *
from mojeUtils import *


SET = 'test' #'test' 'train'
SEQ_NAME = 'ParkingLot2_009_burpeejump1' #'Gym_010_dips1' 'Pavallion_003_018_tossball'
rich_path = 'C:/Users/Dell/inzynierka/rich_toolkit-main'
camera_ids = list(range(8)) #camera ids to be iterated 

with open('C:/Users/Dell/inzynierka/mapping.json') as f:
    mappingOpenpose = json.load(f)

joint_names = get_keypoints()

for camera_id in camera_ids:

    print('Camera ', camera_id)
    print("Start \n")
    gt_keypoints, headCoords, images_found, params_not_found  = getGT(rich_path, SET, SEQ_NAME, camera_id, mappingOpenpose)
    

    #write rich keypoints to pkl files
    save_path = os.path.join(rich_path,'results', SET, SEQ_NAME, f'cam_{camera_id:02d}')
    output = open(save_path +'/keypoints.pkl', 'wb')
    pickle.dump(gt_keypoints, output)
    output.close()

    output = open(save_path +'/heads.pkl', 'wb')
    pickle.dump(headCoords, output)
    output.close()

    output = open(save_path +'/found.pkl', 'wb')#images found in gt
    pickle.dump(images_found, output)
    output.close()

    output = open(save_path +'/not_found.pkl', 'wb')#params not found for images in gt
    pickle.dump(params_not_found, output)
    output.close()
