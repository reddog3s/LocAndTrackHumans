import matplotlib.pyplot as plt

from mojeFunkcjePCh import *
from mojeFunkcjeAP import *
from mojeUtils import *


path = 'C:/Users/PC/inzynierka/dron/bieg1.json'
gt_keypoints, headCoords, images_found = getGtDron(path)


joint_names = get_keypoints()

pred_path = 'C:/Users/PC/inzynierka/videos/pred/dron/bieg1'

predicted_keypoints, images_found = getPred(pred_path)


#whole drone sequence is longer than annotated frames
#if there's more predicted frames than gt frames, crop gt frames
if len(gt_keypoints) != len(predicted_keypoints):
    predicted_keypoints = predicted_keypoints[:len(gt_keypoints)]



joint_dist_list = []
joint_KCP_list = []
ap_list = []
thresholds = np.arange(0.1, 0.6, 0.05)
for threshold in thresholds:
    joint_avg_dist, joint_KCP, no_heads = eval_human_dataset_2d_PCKh(predicted_keypoints,
                                                        gt_keypoints,
                                                        headCoords,
                                                        num_joints=25,
                                                        neck_id=1,
                                                        h_th=threshold,
                                                        iou_th=0.5)

    joint_dist_list.append(joint_avg_dist)
    joint_KCP_list.append(np.average(joint_KCP))

    ap_2D = eval_ap_mpii_v2(predicted_keypoints, [],
                            gt_keypoints, gt_visibility_set=[],
                            headList=headCoords, neck_id=1, joint_names=joint_names, thresh=threshold)
    ap_list.append(ap_2D[-1])



print(joint_KCP_list)
plt.figure(1)
#plt.title('PCKh')
plt.scatter(thresholds, joint_KCP_list)
plt.xlabel('Ułamek długości głowy')
plt.ylabel('Średni wynik PCKh')



plt.figure(2)
#plt.title('mAP')
plt.scatter(thresholds, ap_list)
plt.xlabel('Ułamek długości głowy')
plt.ylabel('mAP')

plt.show()
