import os
import lib.utils.base_utils as util

def makefolders(main_folder ,saving_path):
    # structure of data:
    raw_NDDS_directory = os.path.join(main_folder, 'ndds_captured')

    depth_folder = 'depth'
    mask_folder = 'mask_all'
    mask_folder2 = 'merged_masks'
    rgb_folder = 'rgb'
    valid_poses = 'valid_poses'
    models= 'models'

    main_model_path = util.make_empty_folder(saving_path, models)


    # create folders
    depth_path = util.make_empty_folder(saving_path, depth_folder)
    mask_path = util.make_empty_folder(saving_path, mask_folder)
    mask_path2 = util.make_empty_folder(saving_path, mask_folder2)
    rgb_path = util.make_empty_folder(saving_path, rgb_folder)
    valid_poses_path = util.make_empty_folder(saving_path, valid_poses)

    return depth_path,mask_path,mask_path2,rgb_path,valid_poses_path,main_model_path,raw_NDDS_directory