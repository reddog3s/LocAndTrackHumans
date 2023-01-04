from mojeUtils import *
import cv2

path = 'C:/Users/PC/inzynierka/dron/bieg1.json'
images, headCoords, p = getGtDron(path)

pred_path = 'C:/Users/PC/inzynierka/videos/pred/dron/bieg1'
predicted_keypoints, p = getPred(pred_path)




cap = cv2.VideoCapture('C:/Users/PC/inzynierka/dron/Bieg1.MP4')

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")


count = 0
while cv2.waitKey(33) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    for person in images[count]:
        for keypoint in person:
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 6, (255, 0, 255), thickness=5) #gt fioletowe
        
    for person in predicted_keypoints[count]:
        for keypoint in person:
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 6, (125, 255, 0), thickness=5) #pred zielone


    scale_percent = 40 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame,dim)


    cv2.imshow('dron',frame)
    count+=1

cap.release()
cv2.destroyAllWindows()