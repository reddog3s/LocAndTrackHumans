import cv2
import os
import pickle
from mojeUtils import *


SET = 'test' #'test' 'train'
SEQ_NAME = 'ParkingLot2_009_burpeejump1' #'Gym_010_dips1' 'ParkingLot2_009_burpeejump1' 'Pavallion_003_018_tossball' 'ParkingLot1_004_005_greetingchattingeating1' 'LectureHall_009_021_reparingprojector1'
seq_names = SEQ_NAME.split('_')
SCENE_NAME = seq_names[0]
CAMERA_ID = 1
FRAME_ID = 375



rich_path = 'C:/Users/PC/inzynierka/rich_toolkit-main'
save_path = os.path.join(rich_path,'results', SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}')

#read gt data from pkl files
output = open(save_path +'/keypoints.pkl', 'rb')
gt_keypoints = pickle.load(output)
output.close()

output = open(save_path +'/heads.pkl', 'rb')
headCoords = pickle.load(output)
output.close()

output = open(save_path +'/found.pkl', 'rb')#images found in gt
images_found = pickle.load(output)
output.close()

setInvalidJoints(gt_keypoints, headCoords)


pred_path = os.path.join('C:/Users/PC/inzynierka/videos/pred/rich-dataset', SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}')

predicted_keypoints, images_found = getPred(pred_path, images_found)

#if not all gt keypoints are found in predicted data, delete unmatched keypoints from gt data
for i in list(images_found.values()):
    del gt_keypoints[i]




imgext = json.load(open('resource/imgext.json','r'))
EXT = imgext[SCENE_NAME]

path_to_img_folder = os.path.join('G:/rich-dataset/images', SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}')


img_folder = os.listdir(path_to_img_folder)
#find frame in folder with images
for count, img in enumerate(img_folder):
    if f'{FRAME_ID:05d}_{CAMERA_ID:02d}.{EXT}' == img:
        break

path_to_img = os.path.join(path_to_img_folder, f'{FRAME_ID:05d}_{CAMERA_ID:02d}.{EXT}')
frame = cv2.imread(path_to_img)


for person in gt_keypoints[count]:
    for keypoint in person:
        print((int(keypoint[0]), int(keypoint[1])))
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 10, (255, 0, 255), thickness=-1) #gt purple
    
for person in predicted_keypoints[count]:
    for keypoint in person:
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 10, (125, 255, 0), thickness=-1) #pred green


scale_percent = 30 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
frame = cv2.resize(frame, dim)

save_img_path = os.path.join('C:/Users/PC/inzynierka/sieci/obrazy', SEQ_NAME+'frame%d.png' % FRAME_ID)
print(save_img_path)
cv2.imwrite(save_img_path, frame)
cv2.imshow('Frame', frame)

cv2.waitKey(0)

