import os
import numpy as np

from mojeFunkcjePCh import *
from mojeFunkcjeAP import *
from mojeUtils import *

SET = 'train' #'test' 'train'
SEQ_NAME = 'ParkingLot1_004_005_greetingchattingeating1' #'Pavallion_003_018_tossball' 'ParkingLot1_004_005_greetingchattingeating1' 'LectureHall_009_021_reparingprojector1' 'ParkingLot2_009_burpeejump1' 'Gym_010_dips1'
camera_ids = list(range(8)) #camera ids to be iterated 

joint_KCP_list_all = []
ap_list_all = []

joint_names = get_keypoints()


for camera_id in camera_ids:

    rich_path = 'C:/Users/PC/inzynierka/rich_toolkit-main'
    save_path = os.path.join(rich_path,'results', SET, SEQ_NAME, f'cam_{camera_id:02d}')

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
    human_gt_set_visibility = setVisibility(gt_keypoints)

    pred_path = os.path.join('C:/Users/PC/inzynierka/videos/pred/rich-dataset', SET, SEQ_NAME, f'cam_{camera_id:02d}')

    predicted_keypoints, images_found = getPred(pred_path, images_found)


    #if not all gt keypoints are found in predicted data, delete unmatched keypoints from gt data
    for i in list(images_found.values()):
        del gt_keypoints[i]

    joint_dist_list = []
    joint_KCP_list = []
    ap_list = []
    thresholds = np.arange(0.1, 0.6, 0.05) #0.025
    for threshold in thresholds:
        joint_avg_dist, joint_KCP, no_heads = eval_human_dataset_2d_PCKh(predicted_keypoints,
                                                            gt_keypoints,
                                                            headCoords,
                                                            num_joints=25,
                                                            neck_id=1,
                                                            h_th=threshold,
                                                            iou_th=0.5,
                                                            human_gt_set_visibility = human_gt_set_visibility)

        joint_dist_list.append(joint_avg_dist)
        joint_KCP_list.append(np.average(joint_KCP))

        ap_2D = eval_ap_mpii_v2(predicted_keypoints, [],
                                gt_keypoints, gt_visibility_set=human_gt_set_visibility,
                                headList=headCoords, neck_id=1, joint_names=joint_names, thresh=threshold)
        ap_list.append(ap_2D[-1])

    joint_KCP_list_all.append(joint_KCP_list)
    ap_list_all.append(ap_list)

#write data to files
joint_KCP_list_all = np.asarray(joint_KCP_list_all).transpose()
np.savetxt("PCKthresh_" + SEQ_NAME +".csv", joint_KCP_list_all, delimiter = ",", fmt='%.10f')

ap_list_all = np.asarray(ap_list_all).transpose()
np.savetxt("APthresh_" + SEQ_NAME +".csv", ap_list_all, delimiter = ",", fmt='%.10f')
