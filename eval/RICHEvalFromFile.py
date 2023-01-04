import os
import pickle
import pandas as pd

from mojeFunkcjePCh import *
from mojeFunkcjeAP import *
from mojeUtils import *

SET = 'test' #'test' 'train'
SEQ_NAME = 'ParkingLot2_009_burpeejump1' #'Gym_010_dips1' 'Pavallion_003_018_tossball' 'ParkingLot1_004_005_greetingchattingeating1' 'LectureHall_009_021_reparingprojector1' 'ParkingLot2_009_burpeejump1'
CAMERA_ID = 3

seq_names = SEQ_NAME.split('_')
seq_shortname= seq_names[-1]
seq_shortname = ''.join([i for i in seq_shortname if not i.isdigit()]) #delete digits from seq_name


joint_names = get_keypoints()
rich_path = 'C:/Users/PC/inzynierka/rich_toolkit-main'
save_path = os.path.join(rich_path,'results', SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}')

#read gt data from files
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



print('2d PCKh-0.5\n')
joint_avg_dist, joint_KCP, no_heads = eval_human_dataset_2d_PCKh(predicted_keypoints,
                                                       gt_keypoints,
                                                       headCoords,
                                                       num_joints=25,
                                                       neck_id=1,
                                                       iou_th=0.5)
for i, name in enumerate(joint_names):
    print('     joint: {},  PCK: {:03f}, avg 2D error: {:03f}'.format(name,
                                                                      joint_KCP[i],
                                                                      joint_avg_dist[i]))
print('Invalid frames: ', no_heads[0]/no_heads[1]) #percent of people without heads

print('\n     Overall: PCK: {:03f}, avg 2D error: {:03f} +/- {:03f}\n'.format(np.average(joint_KCP),
                                                                    np.average(joint_avg_dist), np.std(joint_avg_dist)))


#####################################################################################
print('2d mAP\n')
ap = eval_ap_mpii_v2(predicted_keypoints, [],
                        gt_keypoints, gt_visibility_set=[],
                        headList=headCoords, neck_id=1, joint_names=joint_names, thresh=0.5)

for j, name in enumerate(joint_names):
    print('    {},  AP: {:03f}'.format(name, ap[j]))


print('\n     Overall: AP: {:03f} +/- {:03f}\n'.format(ap[-1], np.std(ap[:-1])))


#write to results to csv
results = pd.DataFrame(columns = ["punkt","PCK","AP","AvgErr","nazwa"])
results["punkt"] = joint_names
results["PCK"] = joint_KCP
results["AP"] = ap[:25]
results["AvgErr"] = joint_avg_dist
results["nazwa"] = [SEQ_NAME+str(CAMERA_ID)]*25

results.to_csv('eval/'+seq_shortname+str(CAMERA_ID) +'.csv')
