import os
import json


from mojeFunkcjePCh import *
from mojeFunkcjeAP import *
from mojeUtils import *

SET = 'train' #'test' 'train'
SEQ_NAME = 'Pavallion_003_018_tossball' #'Gym_010_dips1' 'Pavallion_003_018_tossball'
CAMERA_ID = 0
write_to_files = False #if you want to write gt data to pkl files, set to True

rich_path = 'C:/Users/PC/inzynierka/rich_toolkit-main'
save_path = os.path.join(rich_path,'results', SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}')

joint_names = get_keypoints()

with open('C:/Users/PC/inzynierka/mapping.json') as f:
    mappingOpenpose = json.load(f)

gt_keypoints, headCoords, images_found, params_not_found  = getGT(rich_path, SET, SEQ_NAME, CAMERA_ID, mappingOpenpose, save_path)

pred_path = os.path.join('C:/Users/PC/inzynierka/videos/pred/rich-dataset', SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}')

predicted_keypoints, images_found = getPred(pred_path, images_found)


for i in list(images_found.values()):
    del gt_keypoints[i]


print('2d PCKh-0.5\n')
joint_avg_dist, joint_KCP = eval_human_dataset_2d_PCKh(predicted_keypoints,
                                                       gt_keypoints,
                                                       headCoords,
                                                       num_joints=25,
                                                       neck_id=1,
                                                       iou_th=0.5)
for i, name in enumerate(joint_names):
    print('     joint: {},  PCK: {:03f}, avg 2D error: {:03f}'.format(name,
                                                                      joint_KCP[i],
                                                                      joint_avg_dist[i]))

print('\n     Overall: PCK: {:03f}, avg 2D error: {:03f} +/- {:03f}\n'.format(np.average(joint_KCP),
                                                                    np.average(joint_avg_dist), np.std(joint_avg_dist)))


#####################################################################################
print('2d mAP\n')
ap_2d = eval_ap_mpii_v2(predicted_keypoints, [],
                        gt_keypoints, gt_visibility_set=[],
                        headList=headCoords, neck_id=1, joint_names=joint_names, thresh=0.5)

if write_to_files:
    #write rich keypoints to pkl files
    save_path = os.path.join(rich_path,'results', SET, SEQ_NAME, f'cam_{CAMERA_ID:02d}')
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

