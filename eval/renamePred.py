import os
import re

def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

#change names of files with predicted data, to names of images 
SET = 'test' #'test' 'train'
SEQ_NAME = 'ParkingLot2_009_burpeejump1' #'Gym_010_dips1' 'Pavallion_003_018_tossball'

CAMERA_ID = 3

pathIMG = os.path.join('G:/rich-dataset/images', SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}')
#pathIMG = os.path.join('C:/Users/PC/inzynierka/videos/pred/rich-dataset', SET, SEQ_NAME, 'cam_01')
pathPred = os.path.join('C:/Users/PC/inzynierka/videos/pred/rich-dataset', SET, SEQ_NAME, 'cam_02')
images = os.listdir(pathIMG)
predFiles = os.listdir(pathPred)

predFiles.sort(key=num_sort)
images.sort(key=num_sort)

for count, image in enumerate(images):
    frame_id = num_sort(image)
    old_name = os.path.join(pathPred, predFiles[count])
    new_name = os.path.join(pathPred, str(frame_id)+'.json')
    os.rename(old_name, new_name)
