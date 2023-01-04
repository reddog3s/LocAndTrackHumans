from mojeFunkcjePCh import *
from mojeFunkcjeAP import *
from mojeUtils import *

path = 'C:/Users/PC/inzynierka/dron/bieg1.json' #path to annotations
#get keypoints from annotations
gt_keypoints, headCoords, images_found = getGtDron(path)

#get joint names
joint_names = get_keypoints()

pred_path = 'C:/Users/PC/inzynierka/videos/pred/dron/bieg1' #path to folder with predictions

#get predicted keypoints
predicted_keypoints, images_found = getPred(pred_path)

#whole drone sequence is longer than annotated frames
#if there's more predicted frames than gt frames, crop gt frames
if len(gt_keypoints) != len(predicted_keypoints):
    predicted_keypoints = predicted_keypoints[:len(gt_keypoints)]


print('2d PCKh-0.5\n')
joint_avg_dist, joint_KCP, no_heads = eval_human_dataset_2d_PCKh(predicted_keypoints,
                                                       gt_keypoints,
                                                       headCoords,
                                                       num_joints=25,
                                                       neck_id=1,
                                                       iou_th=0.5)

print('Invalid frames: ', (no_heads[0]/no_heads[1])*100) #percent of people without heads

for i, name in enumerate(joint_names):
    print('     joint: {},  PCK: {:03f}, avg 2D error: {:03f}'.format(name,
                                                                      joint_KCP[i],
                                                                      joint_avg_dist[i]))

print('\n     Overall: PCK: {:03f} +/- {:03f}, avg 2D error: {:03f} +/- {:03f}\n'.format(np.average(joint_KCP), np.std(joint_KCP),
                                                                    np.average(joint_avg_dist), np.std(joint_avg_dist)))


#####################################################################################
print('2d mAP\n')

ap = eval_ap_mpii_v2(predicted_keypoints, [],
                        gt_keypoints, gt_visibility_set=[],
                        headList=headCoords, neck_id=1, joint_names=joint_names, thresh=0.5)

for j, name in enumerate(joint_names):
    print('    {},  AP: {:03f}'.format(name, ap[j]))


for i, name in enumerate(joint_names):
    print('{},{},{},{}'.format(name, joint_KCP[i], ap[i], joint_avg_dist[i]))
print('\n     Overall: AP: {:03f} +/- {:03f}\n'.format(ap[-1], np.std(ap[:-1])))