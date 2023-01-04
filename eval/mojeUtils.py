import os
import json
import pickle
import torch
import numpy as np

from smplx import SMPLX
from utils import CalibratedCamera
import re

def num_sort(test_string):
    '''extract int numbers from string'''
    return list(map(int, re.findall(r'\d+', test_string)))[0]

def get_keypoints():
    '''get list of keypoints'''
    keypoints = [
        'Nose',
        'Neck',
        'RShoulder',
        'RElbow',
        'RWrist',
        'LShoulder',
        'LElbow',
        'LWrist',
        'MidHip',
        'RHip',
        'RKnee',
        'RAnkle',
        'LHip',
        'LKnee',
        'LAnkle',
        'REye',
        'LEye',
        'REar',
        'LEar',
        'LBigToe',
        'LSmallToe',
        'LHeel',
        'RBigToe',
        'RSmallToe',
        'RHeel']
    return keypoints

def getPred(path, images_found = None):
    '''
    read predictions from path
    Arguments:
    path - path to folder with predictions in json from OpenPose
    images_found - images which have been found in ground truth (gt) data, optional, list or dict

    Returns:
    all_people - list of predictions in format [frames x people_on_frame x keypoints]
    images_found - return input list/dict of images found in gt data without images which,
                   if all images from gt are found in predicted data, empty list/dict is returned
    '''

    pred_folder = os.listdir(path)
    pred_folder.sort(key=num_sort) #sort files in folder by frame number

    if images_found != None:
        temp = []
        for image in pred_folder:
            if (num_sort(image)) in images_found:
                images_found.pop(num_sort(image))
                temp.append(image)
        pred_folder = temp


    all_people = [] #list of all people on frames

    for image in pred_folder:
        g = open(os.path.join(path, image))
        predicted_file = json.load(g)
        g.close()

        people_on_frame = []
        for i in predicted_file['people']:
            person = np.asarray([i['pose_keypoints_2d'][::3],
                            i['pose_keypoints_2d'][1::3]]).transpose().tolist()
            people_on_frame.append(person)
                

        #openpose writes 0 in coordinates if joint isnt detected
        #change these coordinates from 0 to -1 since 0 is a valid coordinate in image
        for human in people_on_frame:
            for j in human:
                if np.logical_and(j[0] == 0, j[1] == 0): 
                    j[0] = -1
                    j[1] = -1
        all_people.append(people_on_frame)

    return all_people, images_found

def getGtDron(path):
    '''
    read ground truth (gt) for drone data from path
    Arguments:
    path - path to json file with gt data, annotation are in COCO Keypoint format

    Returns:
    people - list of ground truth data in format [frames x people_on_frame x keypoints], to access keypoint coordinates as [x, y] use expression people[frame_number][person_number][keypoint_number]
    headCoords - list of head center coordinates in format [frames x people_on_frame x head_coordinates], to access head coordinates as [x, y] use expression people[frame_number][person_number]
    images_found - images which have been found in ground truth (gt) data, list
    '''

    people = []
    people_visiblities = []
    images_found = []

    with open(os.path.join(path)) as f:
        pred_data = json.load(f)
        
    frame_vis = [] #visibilities of keypoints on frame
    people_on_frame = [] 

    sortedPeople = {} 

    #wrtie annotation to dict in pairs annotation_number : frame_number
    for annotation_number, annotation in enumerate(pred_data['annotations']):
        image_id = annotation['image_id']
        image_name = pred_data['images'][image_id-1]['file_name']
        frame_number = num_sort(image_name)
        sortedPeople.update({annotation_number : frame_number})

    #sort dict with people
    sortedPeople = dict(sorted(sortedPeople.items(), key=lambda item: item[1]))
    #change dict to list of tuples
    sortedPeople = list(sortedPeople.items())

    prev_image_id = sortedPeople[0][1] #set previous image_id as the first one

    for annotation_number, frame_number in sortedPeople:
        annotation = pred_data['annotations'][annotation_number]
        #extract annotation for single person
        person = np.asarray([annotation['keypoints'][::3], annotation['keypoints'][1::3]]).transpose().tolist()

        #append frame number to list of images which have been founded
        images_found.append(frame_number)

        if frame_number == prev_image_id:
            #if current frame number is the same as the previous one, add person to people on frame
            people_on_frame.append(person)
            frame_vis.append(annotation['keypoints'][2::3]) #add visibilities to visibilities on frame
        else:
            #if current frame number isnt the same as the previous one
            people.append(people_on_frame) #add people on frame to list of all people
            people_on_frame = []
            people_on_frame.append(person) #add person to new list of people on frame

            people_visiblities.append(frame_vis) #add visibilities on frame to list of all visibilities
            frame_vis = [] 
            frame_vis.append(annotation['keypoints'][2::3]) #add visibilities to new list of visibilities on frame

            prev_image_id = frame_number #set previous number to current one

    people.append(people_on_frame) #needed for last frame to be added
    people_visiblities.append(frame_vis) 


    #if joint (keypoint) isnt visibile, set its coordinates to -1
    for img_number, image in enumerate(people_visiblities):
        for person_number, person in enumerate(image):
            for joint_number, joint in enumerate(person):
                if joint == 0:
                    people[img_number][person_number][joint_number][0] = -1
                    people[img_number][person_number][joint_number][1] = -1


    headCoords = []
    #extract head coords from data
    for image in people:
        people_heads = []
        for person in image:
            people_heads.append(person.pop())
        headCoords.append(people_heads)

    #take only unique frame numbers
    images_found = set(images_found)
    images_found = (list(images_found))
    images_found.sort()

    return people, headCoords, images_found

def getGT(rich_path, set_name, seq_name, camera_id, mapping, save_path = None):
    '''
    read ground truth (gt) for one sequence from RICH dataset from path
    Arguments:
    rich_path - path to folder with rich-toolkit
    set_name - name of set (test or train)
    seq_name - name of the sequence
    camera_id - id of camera 
    mapping - mapping of OpenPose joints to RICH data, RICH uses SMPL-X model which has all keypoints from OpenPose BODY_25, hand model and face model but indices of the keypoints are not the same as in OpenPose
                it means that the first 25 keypoints, arent the BODY_25 keypoints
                mapping contains relations between SMPL-X and OpenPose keypoints indices,
                mapping used in project has been downloaded from GitHub https://github.com/Meshcapade/wiki/tree/main/assets/SMPLX_OpenPose_mapping
    save_path - path to folder to save RICH data, optional


    Returns:
    gt_keypoints - list of ground truth data in format [frames x people_on_frame x keypoints], to access keypoint coordinates as [x, y] use expression people[frame_number][person_number][keypoint_number]
    headCoords - list of head center coordinates in format [frames x people_on_frame x head_coordinates], to access head coordinates as [x, y] use expression people[frame_number][person_number]
    images_found - images which have been found in ground truth (gt) data, dict in format frame_id:img_index_in_output_list
    params_not_found - list of frames for which parameter arent found in gt, list of tuples in format (frame_id, sub_id)
    '''

    seq_names = seq_name.split('_')
    scene_name = seq_names[0]
    sub_ids = [] #sub_ids of people in sequence, one sub_id corresponds to one person
    for name in seq_names:
        if name.find('0',0,1) !=-1:
            sub_ids.append(name)

    gender_mapping = json.load(open(os.path.join(rich_path, 'resource/gender.json'),'r')) 
    
    smplx_model_dir = os.path.join(rich_path, 'body_models/smplx') #path to SMPL-X model

    gt_params_path = os.path.join(rich_path, 'data/bodies', set_name, seq_name) #path to SMPL-X bodies

    
    img_folder = os.listdir(gt_params_path)
    img_folder.sort(key=num_sort) #sort files in folder by frame number

    params_not_found = []
    image_found = {}
    gt_keypoints = []
    headCoords = [] 

    #iterate images in folder
    for count, image in enumerate(img_folder):
        frame_id = num_sort(image) #exctract frame number from image name
        image_found.update({frame_id:count}) #append frame number to dict
        people = []
        peopleHeads = []
        for person_id in sub_ids: #iterate people on frame

            gender = gender_mapping[f'{int(person_id)}']   

            body_model = SMPLX(
                        smplx_model_dir,
                        gender=gender,
                        num_pca_comps=12,
                        flat_hand_mean=False,
                        create_expression=True,
                        create_jaw_pose=True,
                    )

            smplx_params_fn = os.path.join(rich_path, 'data/bodies', set_name, seq_name, f'{frame_id:05d}', f'{person_id}.pkl')

            #if body parameters arent found for current frame, append that frame_id to list and go to next iteration
            try:
                body_params = pickle.load(open(smplx_params_fn,'rb'))
            except FileNotFoundError:
                params_not_found.append((frame_id, person_id))
                continue


            body_params = {k: torch.from_numpy(v) for k, v in body_params.items()}
            body_model.reset_params(**body_params)
            model_output = body_model(return_verts=True,   
                                    body_pose=body_params['body_pose'],
                                    return_full_pose=True) 

            ## project to image
            calib_path = os.path.join(rich_path, 'data/scan_calibration', scene_name, 'calibration', f'{camera_id:03d}.xml')
            cam = CalibratedCamera(calib_path=calib_path)
            j_2D = cam(model_output.joints).squeeze().detach().numpy()

            #append keypoints coordinates to person
            person = []
            for i in mapping['smplx_idxs'][:25]:
                #print((int(j_2D[i][0]), int(j_2D[i][1])))
                person.append([j_2D[i][0], j_2D[i][1]])
            
            #append head center coordinates of the current person to list of people's heads on frame
            peopleHeads.append([j_2D[15][0], j_2D[15][1]])
            #append person to list of people on frame
            people.append(person)

        #append people's heads on frame to list of head coords
        headCoords.append(peopleHeads) 
        #append people on frame to list of all people
        gt_keypoints.append(people)

        #optionally save data to pkl files
        if save_path != None:
            output = open(save_path +'/keypoints.pkl', 'wb')
            pickle.dump(gt_keypoints, output)
            output.close()

            output = open(save_path +'/heads.pkl', 'wb')
            pickle.dump(headCoords, output)
            output.close()

            output = open(save_path +'/found.pkl', 'wb')#images found in gt
            pickle.dump(image_found, output)
            output.close()

            output = open(save_path +'/not_found.pkl', 'wb')#params not found for images in gt
            pickle.dump(params_not_found, output)
            output.close()

    return gt_keypoints, headCoords, image_found, params_not_found

def setInvalidJoints(gt_keypoints, headCoords):
    '''
    Some keypoints in RICH sequences have their coordinates larger than the image resolution
    it's caused by the fact that some keypoints can be visible from one camera but not from the rest
    all keypoints are read by projection data to camera view, which results in keypoints, which were originally not visible from that camera, having coordinates larger than the resolution
    these coordinates should be set to -1 if they arent in the camera view
    Arguments:
    gt_keypoints - list of ground truth data in format [frames x people_on_frame x keypoints]
    headCoords - list of head center coordinates in format [frames x people_on_frame x head_coordinates]
    '''
    w = 4112 #width of an image
    h = 3008 #height of an image
    
    for image in gt_keypoints:
        for human in image:
            for j in human:
                if np.logical_or(j[0] > w, j[1] > h):
                    j[0] = -1
                    j[1] = -1

    for image in headCoords:
        for human in image:
            if np.logical_or(j[0] > w, j[1] > h):
                j[0] = -1
                j[1] = -1

