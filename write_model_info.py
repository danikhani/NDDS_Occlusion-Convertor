import lib.utils.ply_utils as ply

import os
import shutil
from natsort import natsorted

def export_model_info(yaml_path, ply_path, model_number):
    model_corner, model_distances, model_diameter = ply.read_ply_info(ply_path)
    ply.export_model_para_yml(yaml_path, model_number, model_corner, model_distances,model_diameter)


def export_all_ply_from_folder(folder_path,yml_folder_path,model_info_yaml_path,model_list):

    # remove the old yml file
    try:
        os.remove(model_info_yaml_path)
    except FileNotFoundError:
        print(model_info_yaml_path + ' doesnt exist')

    # list files in folder_path
    sorted_list_of_files = natsorted(os.listdir(folder_path))
    for data_name in sorted_list_of_files:
        source_file = os.path.join(folder_path, data_name)
        for model in model_list:
            if data_name == model['obj_class']+'.ply':
                # copy the ply files to the destination
                new_name = 'obj_{}.ply'.format(model['obj_id'])
                edited_filename = data_name.replace(data_name, new_name)
                new_destination = os.path.join(yml_folder_path, edited_filename)
                shutil.copy2(source_file, new_destination)
                export_model_info(model_info_yaml_path,new_destination,model['obj_id'])

