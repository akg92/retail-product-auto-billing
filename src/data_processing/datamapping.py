import json
from configuration import Configuration
import os, shutil
import utils

conf = Configuration()
original_dir = conf.get_data_dir()
train_dir = conf.get_train_dir()
train_old_dir = conf.get_train_old_dir()

def get_train_json():
    train_json = None
    train_json_file = os.path.join(original_dir , 'instances_train2019.json')
    with open(train_json_file, 'r') as f:
        train_json = json.load(f)

    return train_json



def get_map(json_map):
    id_to_filename = {}
    for obj in json_map['images']:
        id_to_filename[str(obj['id'])] = obj['file_name']
    return json_map['annotations'], id_to_filename


def create_cat_dir(dir, category):
    path = os.path.join(dir, category)
    try:
        os.mkdir(path)
    except:
        os.rmdir(path)
        os.mkdir(path)
        print('{} path cleaned'.join(path))


def reorg_dir(id_to_filename, src, dest):
    found = set()
    for id, file_name in id_to_filename.items():
        if id not in found:
            create_cat_dir(dest, id)
            found.add(id)
        src_file_name = os.path.join(src, file_name)
        dest_file_name = os.path.join(dest, file_name)
        shutil.copy(src_file_name, dest_file_name)

"""
 One time initial mapping call. 
 The files will be populated.
"""
def initial_mapping():
    train_json = get_train_json()
    json_map,id_to_filename = get_map(train_json)
    utils.create_folder_if_not_exist(train_dir)
    reorg_dir(id_to_filename,train_old_dir,train_dir)



