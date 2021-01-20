import os
from natsort import natsorted
import numpy as np

import lib.utils.base_utils as util

def make_linemode_dataset(raw_NDDS_directory,saving_path,scale,train_percentage):
    # read settings.json
    object_list_with_dic = util.read_object_setting_json(raw_NDDS_directory+'/_object_settings.json')
    cam_K, depth_scale, image_size = util.read_camera_intrinsic_json(raw_NDDS_directory+'/_camera_settings.json')

    print('camera.settings.json found and instritics updated:')
    print(cam_K)
    print(depth_scale)
    print(image_size)
    print('object_settings.json found and objects extracted:')
    print(object_list_with_dic)

    # structure of data:
    main_data_path = os.path.join(saving_path, 'data/{}'.format(1))
    main_model_path = os.path.join(saving_path, 'models')

    depth_folder = 'depth'
    mask_folder = 'mask_all'
    mask_folder2 = 'merged_masks'
    rgb_folder = 'rgb'

    # yaml files path
    ground_truth_path = main_data_path + '/gt.yml'
    camera_info_path = main_data_path + '/info.yml'

    # test.txt & train.txt path
    test_txt_path = main_data_path + '/test.txt'
    train_txt_path = main_data_path + '/train.txt'

    #what kind of files want to get extraceted
    depth_ending = '.depth.cm.16.png'
    mask_ending = '.cs.png'
    rgb_ending = '.png'
    all_endings = ['.cs.png','.16.png','.8.png','.micon.png','.depth.png','.is.png']

    # create folder structure
    try:
        os.makedirs(main_data_path)
    except FileExistsError:
        print(main_data_path + ' exist')
    try:
        os.makedirs(main_model_path)
    except FileExistsError:
        print(main_model_path + ' exist')
    # create folders
    depth_path = util.make_empty_folder(main_data_path,depth_folder)
    mask_path = util.make_empty_folder(main_data_path,mask_folder)
    mask_path2 = util.make_empty_folder(main_data_path, mask_folder2)
    rgb_path = util.make_empty_folder(main_data_path,rgb_folder)

    mask_index = 0
    depth_index = 0
    rgb_index = 0
    sorted_list_of_files = natsorted(os.listdir(raw_NDDS_directory))
    for data_name in sorted_list_of_files:
        # find mask data
        if data_name.endswith(mask_ending):
            util.copy_and_rename(raw_NDDS_directory, data_name, mask_index, mask_path)
            util.copy_and_rename(raw_NDDS_directory, data_name, mask_index, mask_path2)
            mask_index += 1
        # find depth data
        if data_name.endswith(depth_ending):
            util.copy_and_rename(raw_NDDS_directory, data_name, depth_index, depth_path)
            depth_index += 1
        # find rgb data
        if data_name.endswith(rgb_ending) and not any(data_name.endswith(x) for x in all_endings):
            util.copy_and_rename(raw_NDDS_directory, data_name, rgb_index, rgb_path)
            rgb_index += 1


    #removing and recreating the yml files:
    try:
        os.remove(ground_truth_path)
    except FileNotFoundError:
        print(ground_truth_path + ' doesnt exist')

    try:
        os.remove(camera_info_path)
    except FileNotFoundError:
        print(camera_info_path + ' doesnt exist')

    # writing the yaml filses:
    yml_index = 0
    for data_name in sorted_list_of_files:
        # writing gt.yml
        if data_name.endswith('.json') and not data_name.endswith('settings.json'):
            # read the json and return dict
            objects_from_annotation = util.read_gt_json(raw_NDDS_directory, data_name)
            #get the arrays from the json files.
            #cam_R_m2c_array, cam_t_m2c_array, obj_bb = util.get_groundtruth_data(objects_from_annotation,scale, object_id)
            obj_index = 0
            gt_info_list = []
            for current_object in objects_from_annotation:
                # check if the indexes match
                if current_object['class'] == object_list_with_dic[obj_index]['obj_class']:
                    # get gt information of current object
                    gt_info_dict = util.get_groundtruth_data(current_object, scale, obj_index)
                    gt_info_list.append(gt_info_dict)
                    obj_index +=1
                else:
                    raise Exception("the index of gt.json and object_settings.json does not match!")

            util.parse_groundtruth_yml(ground_truth_path,yml_index,gt_info_list)

                    #save arrays in gt.yml
            #util.parse_groundtruth_yml(ground_truth_path, yml_index, cam_R_m2c_array, cam_t_m2c_array, obj_bb, object_id)
            # save info.yml
            util.parse_camera_intrinsic_yml(camera_info_path, yml_index, cam_K, depth_scale)
            yml_index +=1

    #generate training and test_files
    training_frames,test_frames = util.make_training_set(0,rgb_index,train_percentage)
    with open(train_txt_path, 'w') as f:
        for item in training_frames:
            f.write("%s\n" % item)

    with open(test_txt_path, 'w') as f:
        for item in test_frames:
            f.write("%s\n" % item)

    print('data generated!')
    print('yml_index,mask_index,depth_index,rgb_index are:')
    print(yml_index,mask_index,depth_index,rgb_index)





