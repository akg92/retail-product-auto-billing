import os
import sys
import  shutil
import json


"""
make directory if not existing

"""
def make_dir(location):
    try:
        os.mkdir(location)
    except:

        print("ignoring the path creation {}".format(location))
        pass

def copy_all_file(src_dir,dest_dir):

    for file in os.listdir(src_dir):
        shutil.copy(os.path.join(src_dir,file),os.path.join(dest_dir,src_dir))


## move some part of test data to the training data for temporary purpose.
## if -1 means move all the records from test dataset.
def mash_up_data(src_dir,dest_dir,n_records=-1):
    test_instance_file = os.path.join(src_dir,'annotations/instances_test2019.json')
    train_instance_file = os.path.join(src_dir,'annotations/instances_test2019.json')
    
    with open(test_instance_file)as f:
        test_json = json.load(f)
    
    with open(train_instance_file)as f:
        train_json = json.load(f)
        
    ## as of now copy all the file.

    train_json['images'] += test_json['images']
    train_json['annotations'] += train_json['annotations']


    make_dir(dest_dir)
    make_dir(os.path.join(dest_dir,"annotations"))
    make_dir(os.path.join(dest_dir,"train2019"))

    ## copy all annotations. The train annotations file will be overwritten.
    copy_all_file(os.path.join(src_dir,"annotations"),os.path.join(dest_dir,"annotations"))

    dest_instance_file = os.path.join(dest_dir, 'annotations/instances_test2019.json')
    ## write joson file
    with open(dest_instance_file) as f:
        json.dump(train_json,f)

    copy_all_file(os.path.join(src_dir,"images/test2019"),os.path.join(dest_dir,"images/train2019"))
    copy_all_file(os.path.join(src_dir, "images/train2019"), os.path.join(dest_dir, "images/train2019"))

    ## create syslink for mainintaining the same directory structure.
    os.symlink(os.path.join(src_dir,"images/test2019"),os.path.join(dest_dir,"images/test2019"))
    os.symlink(os.path.join(src_dir, "images/val2019"), os.path.join(dest_dir, "images/val2019"))




if __name__ == '__main__':

    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    mash_up_data(src_dir,dest_dir)
