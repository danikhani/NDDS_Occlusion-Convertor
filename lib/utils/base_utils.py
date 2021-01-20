import os
import io
import shutil
from scipy.spatial.transform import Rotation as rotate
import numpy as np
import json
import yaml
import random


def make_empty_folder(main_path,folder):
    folder_path = os.path.join(main_path, folder)
    try:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    except FileNotFoundError:
        os.mkdir(folder_path)
    return folder_path

def copy_and_rename(raw_data_directory,picturename,index,destination_path):
    source_file = os.path.join(raw_data_directory, picturename)
    mask_name = '{}.png'.format(index)
    edited_filename = picturename.replace(picturename, mask_name)
    new_destination = os.path.join(destination_path, edited_filename)
    shutil.copy2(source_file, new_destination)

def parse_camera_intrinsic_yml(file_path, number_of_frame, cam_K_array, depth_scale):
    camera_intrinsic = {
        number_of_frame: {
            'cam_K': cam_K_array,
            'depth_scale': depth_scale,
        }
    }
    with io.open(file_path, 'a', encoding='utf8') as outfile:
        yaml.dump(camera_intrinsic, outfile, default_flow_style=None,width=1000)

# {0: [{'cam_R_m2c': [0.0963063, 0.99404401, 0.0510079, 0.57332098, -0.0135081, -0.81922001, -0.81365103, 0.10814, -0.57120699],
# 'cam_t_m2c': [-105.3577515, -117.52119142, 1014.8770132], 'obj_bb': [244, 150, 44, 58], 'obj_id': 1}], 1: ... }
def parse_groundtruth_yml(ground_truth_path, yml_index, gt_info_list ):
    ground_truth = {
        yml_index: gt_info_list
    }
    with io.open(ground_truth_path, 'a', encoding='utf8') as outfile:
        yaml.dump(ground_truth, outfile, default_flow_style=None,width=1000)


def read_camera_intrinsic_json(file_path):
    # https://github.com/thodan/sixd_toolkit/blob/master/doc/sixd_2017_datasets_format.md
    # cam_K - 3x3 intrinsic camera matrix K (saved row-wise)
    # https://github.com/zju3dv/clean-pvnet/issues/118:
    # [fx, 0, ux, 0, fy, uy, 0, 0, 1]
    with open(file_path, 'r') as file:
        annotation = json.loads(file.read())

    camera_settings = annotation['camera_settings']
    intrinsic_settings = camera_settings[0]['intrinsic_settings']
    captured_image_size = camera_settings[0]['captured_image_size']
    # intrinsic matrix
    fx = float(intrinsic_settings['fx'])
    ux = float(intrinsic_settings['cx'])
    fy = float(intrinsic_settings['fy'])
    uy = float(intrinsic_settings['cy'])
    s = float(intrinsic_settings['s'])
    cam_K = [fx, 0, ux, 0, fy, uy, 0, 0, 1.0]
    depth_scale = 1.0
    # image size:
    image_size = [captured_image_size['width'], captured_image_size['height']]

    return cam_K, depth_scale, image_size


def read_object_setting_json(file_path):
    with open(file_path, 'r') as file:
        annotation = json.loads(file.read())
    # list containing dictionary with class data
    object_list_with_dic = []
    object_classes = annotation['exported_objects']
    # iterate over all classes in this file and add them to dictionary
    object_id = 1
    for object in object_classes:
        obj_dict = {
            "obj_id": object_id,
            "obj_class": object['class'],
            "obj_intensity_segmentation": object['segmentation_class_id'],
        }
        object_list_with_dic.append(obj_dict)
        object_id +=1
    return object_list_with_dic


def read_gt_json(raw_data_directory,json_file):
    source_file = os.path.join(raw_data_directory, json_file)
    with open(source_file, 'r') as file:
        annotation = json.loads(file.read())

    objects_from_annotation = annotation['objects']
    return objects_from_annotation

def read_yml_file(folder,yml_file_name):
    source_file = os.path.join(folder, yml_file_name)
    with open(source_file, 'r') as file:
        annotation = yaml.load(file)

    return annotation


def get_groundtruth_data(objects_from_annotation,scale,object_id):

    # translation
    translation = np.array(objects_from_annotation['location']) * scale

    # rotation
    rotation = np.asarray(objects_from_annotation['pose_transform'])[0:3, 0:3]
    rotation = np.dot(rotation, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    rotation = np.dot(rotation.T, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    r1 = rotate.from_euler('z', 90, degrees=True)
    rotation = np.dot(rotation, r1.as_dcm())

    rotation_list = list(rotation[0, :]) + list(rotation[1, :]) + list(rotation[2, :])

    # get bounding box:
    bounding_box = objects_from_annotation['bounding_box']
    xmin,ymin = np.array(bounding_box['top_left'])
    xmax,ymax = np.array(bounding_box['bottom_right'])
    # round the pixels
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    deltax= xmax-xmin
    deltay = ymax - ymin
    obj_bb =[ymin, xmin,deltay,deltax]

    # yaml.dump doenst support numpy arrays. Here they are converted to python array
    cam_R_m2c_array =[]
    cam_t_m2c_array = []
    for r in rotation_list:
        cam_R_m2c_array.append(float(r))

    for t in translation:
        cam_t_m2c_array.append(float(t))

    gt_info_dict = {
        "cam_R_m2c" : cam_R_m2c_array,
        "cam_t_m2c" : cam_t_m2c_array,
        "obj_bb" : obj_bb,
        "obj_id": object_id,
    }
    info_for_txt = {
        'rotation' : cam_R_m2c_array,
        'obj_id': object_id,
        'center' : cam_t_m2c_array
    }

    return gt_info_dict, info_for_txt


# generate test and train data numbers:
def make_training_set(start,end,training_percent):
    # Generate 'n' unique random numbers within a range
    number_of_test = int((1-training_percent)*(end-start))
    test_frames = random.sample(range(start, end), number_of_test)
    test_frames.sort()
    training_frames = list(range(start, end))
    # removing test_frames from the trainings_frame
    for i in test_frames:
        if i in training_frames:
            training_frames.remove(i)

    return training_frames,test_frames

# write the valid_poses.txt files
def parse_validpose_text(text_address, current_frame, image_size, info_for_txt, ply_info):
    file = open(text_address, "w")

    file.writelines('image size\n')
    file.writelines('{} {}\n'.format(image_size[0],image_size[1]))

    file.writelines('{}\n'.format(info_for_txt['obj_id']))

    file.writelines('rotation:\n')
    file.writelines('{} {} {}\n'.format(-info_for_txt['rotation'][1],info_for_txt['rotation'][2],-info_for_txt['rotation'][0]))
    file.writelines(
        '{} {} {}\n'.format(info_for_txt['rotation'][4], -info_for_txt['rotation'][5], info_for_txt['rotation'][3]))
    file.writelines(
        '{} {} {}\n'.format(info_for_txt['rotation'][7], -info_for_txt['rotation'][8], info_for_txt['rotation'][6]))

    file.writelines('center:\n')
    file.writelines(
        '{} {} {}\n'.format(info_for_txt['center'][0]/1000, -info_for_txt['center'][1]/1000, -info_for_txt['center'][2]/1000))

    file.writelines('extend:\n')
    size_y = ply_info['size_y'] / 1000
    size_z = ply_info['size_z'] / 1000
    size_x = ply_info['size_x'] / 1000
    file.writelines('{} {} {}\n'.format(size_y, size_z, size_x))

    file.writelines('\n')
    file.writelines('\n')
    file.writelines('{}\n'.format(current_frame))

    file.close()

def prase_dictionary_yml(file_path, info_dict):

    with io.open(file_path, 'w', encoding='utf8') as outfile:
        yaml.dump(info_dict, outfile, default_flow_style=None,width=1000)

    return True