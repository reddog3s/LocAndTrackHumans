from mojeUtils import *
import cv2
import os

frame_number = 1480
path_to_img = f'G:/bieg1/frame{frame_number}.png'

path = 'C:/Users/PC/inzynierka/dron/bieg1.json'
gt_keypoints, headCoords, p = getGtDron(path)

pred_path = 'C:/Users/PC/inzynierka/videos/pred/dron/bieg1'
predicted_keypoints, p = getPred(pred_path)


frame = cv2.imread(path_to_img)

for person in gt_keypoints[frame_number]:
    for keypoint in person:
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 10, (255, 0, 255), thickness=-1) #gt fioletowe
    
for person in predicted_keypoints[frame_number]:
    for keypoint in person:
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 10, (125, 255, 0), thickness=-1) #pred zielone


scale_percent = 40 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
frame = cv2.resize(frame, dim)

save_path = os.path.join('C:/Users/PC/inzynierka/sieci/obrazy', 'dron_frame%d.png' % frame_number)
cv2.imwrite(save_path, frame)
cv2.imshow('Frame', frame)
cv2.waitKey(0)