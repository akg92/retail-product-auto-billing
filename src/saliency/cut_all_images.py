# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from algorithm import io, np, sk, da
from algorithm import show, random, listdir, mark_boundaries, loga
from algorithm import getSlic, readImg
import algorithm as alg
import skimage as sk
a = random(3, 5)
G = {}  # G is a global var to save value for DEBUG in funcation


IMG_DIR = r'test/'
COARSE_DIR = 'test/'

IMG_NAME_LIST = filter(lambda x: x[-3:] == 'jpg', listdir(IMG_DIR))


imgName = IMG_NAME_LIST[0]
img, imgGt = readImg(imgName)
rgbImg = img

rgbImg = np.zeros((100, 100, 3))
rgbImg[25:75, 25:75, 1:] = 1.


import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

def refinemask(img):
    pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=img, chdim=2)
    img_en = pairwise_energy.reshape((-1, img.shape[0], img.shape[1]))  # Reshape just for plotting
    print(img_en.shape)
    return img_en[-1]

def get_bbox_indexes(img,bounding_box):

    new_bbox = [max(0,int(bounding_box[1]-0.4*bounding_box[3])),int(min(img.shape[1],bounding_box[1]+1.4*bounding_box[3])),
           max(0,int(bounding_box[0]-0.4*bounding_box[2])),int(min(img.shape[0],bounding_box[2]*1.4+bounding_box[0]))]


    old_bbox = [bounding_box[1]-max(0, int(bounding_box[1] - 0.4 * bounding_box[3])),
     int(min(img.shape[1], bounding_box[1] + 1.4 * bounding_box[3]))-bounding_box[1],
     bounding_box[0]-max(0, int(bounding_box[0] - 0.4 * bounding_box[2])),
     int(min(img.shape[0], bounding_box[2] * 1.4 + bounding_box[0]))-bounding_box[0]]
    old_bbox = [int(x) for x in old_bbox]

    return new_bbox,old_bbox

def remove_added_background(img,old_bbox):
    img[0:old_bbox[0],:,:] =0
    img[old_bbox[1]:, :, :] = 0
    img[:,0:old_bbox[2] , :] = 0
    img[:,old_bbox[3]:, :] = 0

def remove_continuos_black(img):

    i_start,i_end,j_start,j_end = 0,img.shape[0]-1,0,img.shape[1]-1

    while(not np.any(img[i_start,:]) and i_start<i_end):
        i_start +=1

    while (not np.any(img[i_end,:]) and i_start < i_end):
        i_end -= 1

    while (not np.any(img[:,j_start]) and j_start < j_end):
        j_start += 1
    while (not np.any(img[:,j_end]) and j_start < j_end):
        j_end -= 1

    return i_start,i_end,j_start,j_end




def extract_image(img_name,bounding_box):
    img = cv2.imread(img_name)
    ## crop the image
    new_bbox,old_bbox = get_bbox_indexes(img,bounding_box)
    img  = img[new_bbox[0]:new_bbox[1],new_bbox[2]:new_bbox[3],]
    build_methods = ['BG1']
    #cv2.imwrite('hellp.jpg',img)
    generated_map = buildImgs_single_from_array(img, build_methods, [])
    generated_multi = np.zeros((img.shape[0],img.shape[1],3))
    generated_multi[:,:,0] = generated_map
    generated_multi[:, :, 1] = generated_map
    generated_multi[:, :, 2] = generated_map
    remove_added_background(generated_multi,old_bbox)

    refined_map = refinemask(generated_multi)
    refined_map = refined_map.astype('uint8')
    ret3, th4 = cv2.threshold(refined_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #bit = cv2.bitwise_and(img, img, mask=th4)
    #bit = remove_continuos_black(bit)

    #remove_added_background(img,old_bbox)
    i_start,i_end,j_start,j_end =remove_continuos_black(th4)
    bit = img[i_start:i_end,j_start:j_end,]

    return bit




from pycocotools.coco import  COCO

def load_annotations(file_name):
    annotaion_data = COCO(file_name)
    return  annotaion_data
def create_temp_file_name(id,src_file,dest_dir):
    camera = os.path.basename(src_file).split("_")[1]
    return  os.path.join(dest_dir,str(id)+"_"+camera)

def get_bounding_box(id,annotation):
    annotation_id = annotation.getAnnIds(imgIds=[id])[0]
    bounding_box = annotation.anns[annotation_id]['bbox']
    return bounding_box

def threaded_ex(id,img_file,img_path,bounding_box):
    save_path = create_temp_file_name(id, img_file, dest_dir)
    ## added extra check
    if os.path.exists(img_path) and not os.path.exists(save_path):
        img = extract_image(img_path, bounding_box)
        #print(save_path)
        cv2.imwrite(save_path, img)
import threading
import thread
import time
from multiprocessing import Process





import os
def do_preprocessing_all_image_multi_processing(start,end,annotation_file,src_dir,dest_dir):
    annotation = load_annotations(annotation_file)

    i = 0
    th_count = 6
    all_annotations = annotation.imgs.items()
    size = len(all_annotations)
    for i in range(start,end):
        if i>=size:
            break
        id = all_annotations[i][0]
        ant = all_annotations[i][1]
        print(str(id)+":"+str(ant))
        img_file = ant["file_name"]
        img_path  = os.path.join(src_dir,img_file)
        #while (threading.active_count()>=th_count):
        #   time.sleep(1)

        bounding_box = get_bounding_box(id, annotation)
        threaded_ex(id,img_file,img_path,bounding_box)
        #th = threading.Thread(target=threaded_ex,args=(id,img_file,img_path,bounding_box))
        #th.start()
        #    time.sleep(100)
        #    return

    #time.sleep(300)

def spin_off_multi_processing(annotation_file,src_dir,dest_dir,splits=16):
    annotation = load_annotations(annotation_file).imgs.items()
    size = len(annotation)
    step = int(size/splits)
    process_ids = []
    #step = 1 ##test
    del annotation
    for i in range(0,size,step):
        p = Process(target=do_preprocessing_all_image_multi_processing,args=(i,i+step, annotation_file, src_dir, dest_dir))
        p.start()
        process_ids.append(p)
        if i == 2:
            break
    ## wait for execution to complete
    for p in process_ids:
        p.join()




def do_preprocessing_all_image(annotation_file,src_dir,dest_dir):
    annotation = load_annotations(annotation_file)

    i = 0
    th_count = 6
    for id,ant in annotation.imgs.items():
        print(str(id)+":"+str(ant))
        img_file = ant["file_name"]
        img_path  = os.path.join(src_dir,img_file)
        #while (threading.active_count()>=th_count):
        #   time.sleep(1)

        bounding_box = get_bounding_box(id, annotation)
        threaded_ex(id,img_file,img_path,bounding_box)
        #th = threading.Thread(target=threaded_ex,args=(id,img_file,img_path,bounding_box))
        #th.start()
        #i+=1
        #if  i%10:
        #    time.sleep(100)
        #    return

    time.sleep(300)







if __name__ == "__main__":
    from algorithm import *

    coarseMethods = []

    buildMethods = ['BG1']
    #buildMethods=['MY1']
    #    num = len(IMG_NAME_LIST)
    #    num = 0
    # print("img list")
    # print(IMG_NAME_LIST)
    # for name in IMG_NAME_LIST[::]:
    #     orginal_img = cv2.imread(os.path.join(IMG_DIR,name))
    #     generated_map = buildImgs_single(name, buildMethods, coarseMethods)
    #     cv2.imwrite('modified_map_adaptive' + name, generated_map)
    #     th,threh_map = cv2.threshold(generated_map, 127, 255, cv2.THRESH_BINARY)
    #     ret3, th4 = cv2.threshold(generated_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     cv2.imwrite('modified_map_th4' + name, th4)
    #     cv2.imwrite('modified_map_'+name,threh_map)
    #     print(generated_map)
    #     bit = cv2.bitwise_and(orginal_img,orginal_img,mask=th4)
    #     cv2.imwrite('anded_' + name, bit)
    #     bit = cv2.bitwise_and(orginal_img,orginal_img,mask=threh_map)
    #     cv2.imwrite('anded_th' + name, bit)
    #
    #     generated_multi = np.zeros((generated_map.shape[0],generated_map.shape[1],3))
    #
    #     generated_multi[:,:,0] = generated_map
    #     generated_multi[:, :, 1] = generated_map
    #     generated_multi[:, :, 2] = generated_map
    #
    #     refined_map = refinemask(generated_multi)
    #     refined_map = refined_map.astype('uint8')
    #     #refined_map = cv2.cvtColor(refined_map, cv2.COLOR_BGR2GRAY)
    #     cv2.imwrite('refined_map_q'+name,refined_map)
    #     print(refined_map.shape)
    #     print(refined_map)
    #     ret3, th4 = cv2.threshold(refined_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     cv2.imwrite('refined_modified_map_th4' + name, th4)
    #     bit = cv2.bitwise_and(orginal_img,orginal_img,mask=th4)
    #     cv2.imwrite('anded_th_new' + name, bit)

    src_dir = '/media/shibin/disk/train2019_/train2019'
    annotation_file = '/media/shibin/disk/train2019_/instances_train2019.json'
    dest_dir  = '/media/shibin/disk/train2019_new'
    #do_preprocessing_all_image(annotation_file,src_dir,dest_dir)
    spin_off_multi_processing(annotation_file,src_dir,dest_dir)
























