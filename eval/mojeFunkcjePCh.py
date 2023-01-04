import numpy as np

def setVisibility(gt_keypoints, headList):
    '''
    set visibility of joints (keypoints) to 0 (not visible) if their coordinates are equal to -1

    Arguments:
    gt_keypoints - list of ground truth data in format [frames x people_on_frame x keypoints]
    headList - list of head center coordinates in format [frames x people_on_frame x head_coordinates]

    Returns:
    human_gt_set_visibility - visibility of keypoints in list, format [frames x people_on_frame x keypoints]
    no_heads - number of people without heads
    '''
    no_heads = 0 
    human_gt_set_visibility = []

    for img_idx, image in enumerate(gt_keypoints):
        visibilites_image = [] #visibilities of keypoints on image
        humans_heads = headList[img_idx] # head coordinates of people on frame
        for human_idx, human in enumerate(image):
            human_vis = [] #visibilities of current person's keypoints
            head = humans_heads[human_idx] # head coordinates of current person
            #if head coordinates are equal to -1, set all joints visibility to 0 in order to avoid errors in evaluation
            if head[0] == -1 and head[1] == -1:
                no_heads +=1
                for joint in human:
                    human_vis.append(0)
            else:
                for joint in human:
                    if joint[0] == -1 and joint[1] == -1:
                        human_vis.append(0)
                    else:
                        human_vis.append(1)

            visibilites_image.append(human_vis)
        human_gt_set_visibility.append(visibilites_image)
     
    return human_gt_set_visibility, no_heads


def compute_bbox_from_humans(humans):
    """
    ATTENTION: only those valid joints are used to calculate bbox
    Arguments:
    humans - list of humans on frame in format [people x keypoints]
    
    Return:
    bboxes - numpy array of bounding boxes for people on frame
    """
    bboxes = []
    for human in humans:
        valid_joints = np.array([joint for joint in human if joint != [-1, -1]])
        if len(valid_joints) == 0:
            return np.array([])
        xmin = np.min(valid_joints[:, 0])
        ymin = np.min(valid_joints[:, 1])
        xmax = np.max(valid_joints[:, 0])
        ymax = np.max(valid_joints[:, 1])
        bboxes.append([xmin, ymin, xmax, ymax])
    return np.array(bboxes)


def bbox_ious(boxes1, boxes2):
    """
    Compute intersection over union (iou) for predicted and gt bounding boxes
    Arguments:
    :param boxes1: N1 X 4, [xmin, ymin, xmax, ymax]
    :param boxes2: N2 X 4, [xmin, ymin, xmax, ymax]

    Return:
    iou - list, N1 X N2
    """
    #if boxes1 is empty, return array of -1, length of boxes2
    if len(boxes1) == 0:
        return np.ones([len(boxes2), 1]) * (-1)

    #if boxes2 is empty, return array of -1, length of boxes1
    if len(boxes2) == 0:
        return np.ones([len(boxes1), 1]) * (-1)
    b1x1, b1y1 = np.split(boxes1[:, :2], 2, axis=1) #min x min y from bbox1
    b1x2, b1y2 = np.split(boxes1[:, 2:4], 2, axis=1) #max x max y from bbox1
    b2x1, b2y1 = np.split(boxes2[:, :2], 2, axis=1) #min x min y from bbox2
    b2x2, b2y2 = np.split(boxes2[:, 2:4], 2, axis=1) #max x max y from bbox2

    dx = np.maximum(np.minimum(b1x2, np.transpose(b2x2)) - np.maximum(b1x1, np.transpose(b2x1)), 0) #max (min from max(1,2) - max from min(1,2)) x
    dy = np.maximum(np.minimum(b1y2, np.transpose(b2y2)) - np.maximum(b1y1, np.transpose(b2y1)), 0) #max (min from max(1,2) - max from min(1,2)) y
    intersections = dx * dy # compute area of rectangle (intersections)

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1) #max x1 - min x1 * max y1 - min y1, compute area of rectangle (bbox)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + np.transpose(areas2)) - intersections #unions is equal to sum of areas minus intersection

    return intersections / unions



def compute_head_size(humans, ind1, headList):
    """
        use the euclidean distance between head center and neck as head size
    Arguments:
        humans: list of people on frame
        ind1: neck index
        headList - list of people's heads on frame

    Returns:
    hsz_vec - list of head sizes for humans

    """
    hsz_vec = []
    for count, human in enumerate(humans):
        hsz_vec.append(np.sqrt((human[ind1][0] -  headList[count][0])**2 + (human[ind1][1] -  headList[count][1])**2))
    return hsz_vec



def match_humans_2d(humans_pred, humans_gt, iou_th=0.5):
    """
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint.
    If the whole ground-truth person got no match, its corresponding joint dists are assigned to be -1.

    ATTENTION:
    1. Assume gt list is not empty, but pred list can be empty.
    2. Those invalid joint due to occlusion lead to invalid distance -1

    :param humans_pred: list of list x, y pos
    :param humans_gt: list of list x, y pos
    :return: a list of K-length array recording joint distances per ground-truth human
    """
    joint_dists = []
    # if prediction is empty, return -1 distance for each gt joint
    if len(humans_pred) == 0:
        for human_gt in humans_gt:
            joint_dists.append(np.ones(len(human_gt))*(-1))
        return joint_dists

    # compute bbox
    bboxes_gt = compute_bbox_from_humans(humans_gt)
    bboxes_pred = compute_bbox_from_humans(humans_pred)

    # compute ious
    ious = bbox_ious(bboxes_gt, bboxes_pred)

    # compute matched joint distance per ground-truth human
    for i, human_gt in enumerate(humans_gt):
        if np.max(ious[i, :]) < iou_th:
            joint_dists.append(np.ones(len(human_gt))*(-1)) #jesli najlepszy wynik jest mniejszy niz prog to nikt nie zostal dopasowany
            continue

        human_pred = humans_pred[np.argmax(ious[i, :])] #indeks tego czlowieka dla ktorego iou jest najwieksze
        human_gt = np.array(human_gt)
        human_pred = np.array(human_pred)
        joint_dist = np.sqrt(np.sum((human_gt-human_pred)**2, axis=1)) #oblicz dystans dla tegi czlowieka
        # invalid detected joint leads to invalid distance -1
        joint_dist[np.logical_and(human_pred[:, 0] == -1, human_pred[:, 1] == -1)] = -1

        joint_dist[np.logical_and(human_gt[:, 0] == -1, human_gt[:, 1] == -1)] = -1 #jesli dany staw gt jest niewidoczny to ustaw go na 0

        # if np.sum(np.logical_and(human_pred[:, 0] == -1, human_pred[:, 1] == -1)) > 0:
        #     print('find invalid joint')
        joint_dists.append(joint_dist)

    return joint_dists




def eval_human_dataset_2d_PCKh(humans_pred_set, humans_gt_set, headList, neck_id, num_joints=25, h_th=0.5, iou_th=0.5, human_gt_set_visibility=None):
    """
    Evaluation of the full dataset by matching predicted humans to ground-truth humans.
    This evaluation considers multi-to-multi matching, although itop dataset only includes single person.
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint. If the distance is < dist_th, the gt joint is
    considered detected.

    Overall metric: average Keypoint Correct Percentage (KCP) over all the joints and over the whole test set

    ATTENTION: those missed detection due to occlusion from other body parts are not treated separately. If the guess is wrong, it's punished.


    :param humans_pred_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param humans_gt_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :head_id: index for head
    :neck_id: index for the connected node
    :param num_joints: number of joints per person
    :param dist_th: the distance threshold in pixels
    :param iou_th: the iou threshold considering two human poses overlapping
    :return:
        joint_avg_dist: average distance per joint for those matched ground-truth joints only
        joint_KCP: KCP per joint over the test set
        (no_heads, number_of_people): no_heads - number of people without heads on all frames, number_of_people - number of people on all frames
    """

    assert len(humans_gt_set) == len(humans_pred_set) 

    if human_gt_set_visibility is None:
        human_gt_set_visibility, no_heads = setVisibility(humans_gt_set, headList)


    human_gt_set_visibility_all = []
    samples_cnt = 0  # number of humans, not number of images
    joint_dists_set = []
    hit_vec = []
    no_head = 0
    for i in range(len(humans_gt_set)):
        humans_gt = humans_gt_set[i]
        humans_pred = humans_pred_set[i]
        samples_cnt += len(humans_gt)

        if len(humans_gt) == 0:
            continue


        joint_dists = match_humans_2d(humans_pred, humans_gt, iou_th)
        hsz_vec = compute_head_size(humans_gt, neck_id, headList[i])

        if human_gt_set_visibility is not None:
            for j, human_gt_visibility in enumerate(human_gt_set_visibility[i]): #iterate visibilities of people from frame
                human_gt_set_visibility_all.append(human_gt_visibility) #append visibility of human to list
                joint_dists[j][np.array(human_gt_visibility) == 0] = -1 #set joint distance to -1 if visibility of that joint is 0
                hit_vec.append(np.logical_and(joint_dists[j] >= 0, joint_dists[j] < hsz_vec[j]*h_th)) #append hits for human to hit_vec list
        joint_dists_set += joint_dists

    human_gt_set_visibility_all = np.array(human_gt_set_visibility_all)

    joint_dists_set = np.array(joint_dists_set)
    number_of_people = len(hit_vec)
    hit_vec = np.array(hit_vec) # [all_people : joints]
    joint_avg_dist = []
    joint_KCP = []
    for k in range(num_joints):
        single_joint_dists = joint_dists_set[:, k]
        joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))

        hit_cnt = np.sum(hit_vec[:, k])
        # A option to only consider gt visible parts
        if human_gt_set_visibility_all.shape[0] is not 0:
            joint_KCP.append((hit_cnt / np.sum(human_gt_set_visibility_all[:, k]))*100)
        else:
            joint_KCP.append((hit_cnt / samples_cnt)*100)

    return joint_avg_dist, joint_KCP, (no_heads, number_of_people) 