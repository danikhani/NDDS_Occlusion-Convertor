import os
from natsort import natsorted

import lib.utils.base_utils as util
import write_model_info as model_info


def make_linemode_dataset(main_folder,saving_path,scale,train_percentage):

    # structure of data:
    raw_NDDS_directory = os.path.join(main_folder,'ndds_captured')
    main_data_path = saving_path
    main_model_path = os.path.join(saving_path, 'models')
    model_info_yaml_path = os.path.join(main_model_path, 'models_info.yml')

    # read settings.json
    object_list_with_dic = util.read_object_setting_json(raw_NDDS_directory + '/_object_settings.json')
    cam_K, depth_scale, image_size = util.read_camera_intrinsic_json(raw_NDDS_directory + '/_camera_settings.json')


    model_info.export_all_ply_from_folder(main_folder + '/ply_files', main_model_path, model_info_yaml_path, object_list_with_dic )
    model_info_yaml_dic = util.read_yml_file(main_model_path,'models_info.yml')


    depth_folder = 'depth'
    mask_folder = 'mask_all'
    mask_folder2 = 'merged_masks'
    rgb_folder = 'rgb'
    valid_poses = 'valid_poses'

    # yaml files path
    ground_truth_path = main_data_path + '/gt.yml'
    camera_info_path = main_data_path + '/info.yml'
    object_segmentation_path = main_data_path + '/objects_segmentation_info.yml'

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
    valid_poses_path = util.make_empty_folder(main_data_path, valid_poses)
    for objects in object_list_with_dic:
        paths = util.make_empty_folder(valid_poses_path, objects['obj_class'])

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
            obj_index = 1
            gt_info_list = []
            for current_object in objects_from_annotation:
                # check if the indexes match
                if current_object['class'] == object_list_with_dic[obj_index-1]['obj_class']:
                    # get gt information of current object
                    gt_info_dict,info_for_txt = util.get_groundtruth_data(current_object, scale, obj_index)
                    gt_info_list.append(gt_info_dict)

                    util.parse_validpose_text(valid_poses_path+'/'+current_object['class']+ '/{}.txt'.format(yml_index), yml_index, image_size, info_for_txt,model_info_yaml_dic[obj_index])
                    obj_index +=1
                else:
                    raise Exception("the index of gt.json and object_settings.json does not match!")
            util.parse_groundtruth_yml(ground_truth_path,yml_index,gt_info_list)

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

    util.prase_dictionary_yml(object_segmentation_path,object_list_with_dic)
    print('data generated!')
    print('yml_index,mask_index,depth_index,rgb_index are:')
    print(yml_index,mask_index,depth_index,rgb_index)





