"""
Generate checkout image from the images from single images.
"""

"""
    cut down the image with bounding box first and get salinity. 
"""
from PIL import  Image
import cv2
import os
import numpy as np
from pycocotools.coco import COCO
import  matplotlib.pyplot as plt


def load_annotations(file_name):
    annotaion_data = COCO(file_name)
    return  annotaion_data


def get_bounding_box(id,annotation):
    annotation_id = annotation.getAnnIds(imgIds=[id])[0]
    bounding_box = annotation.anns[annotation_id]['bbox']
    return bounding_box[0],bounding_box[1],bounding_box[0]+bounding_box[2],bounding_box[1]+bounding_box[3]

def crop_image(image_file,x_l,y_l,x_t,y_t):
    img = Image.open(image_file)
    img = img.crop((x_l,y_l,x_t,y_t))
    print(img.size)
    print('{} {} {} {}'.format(x_l,y_l,x_t,y_t))
    img.save('/home/shibin/Downloads/temp.jpg')
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def largest_contours_rect(saliency):
    im2, contours, hierarchy= cv2.findContours(saliency * 1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = cv2.contourArea)
    return cv2.boundingRect(contours[-1])

def refine_saliency_with_grabcut(img, saliency):
    rect = largest_contours_rect(saliency)
    saliency[np.where(saliency > 0)] = cv2.GC_FGD
    mask = saliency
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return mask

def reduce_bounding_box(img):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 200, 255,cv2.THRESH_BINARY )[1]

    img = cv2.bitwise_and(img,img,mask=threshMap)
    #cv2.imwrite('/home/shibin/Downloads/mask.jpg',threshMap)
    #cv2.imwrite('/home/shibin/Downloads/mask_saliency.jpg', saliencyMap.astype("uint8"))
    return img

def backproject(source, target, levels = 2, scale = 1):
    hsv = cv2.cvtColor(source,  cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        # calculating object histogram
    roihist = cv2.calcHist([hsv],[0, 1], None, \
            [levels, levels], [0, 180, 0, 256] )

        # normalize histogram and apply backprojection
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256], scale)
    return dst


def test_new_method(img):

    backproj = np.uint8(backproject(img, img, levels = 2))
    cv2.normalize(backproj,backproj,0,255,cv2.NORM_MINMAX)
    saliencies = [backproj, backproj, backproj]
    saliency = cv2.merge(saliencies)
    cv2.pyrMeanShiftFiltering(saliency, 20, 200, saliency, 2)
    saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(saliency, saliency)
    (T, saliency) = cv2.threshold(saliency, 200, 255, cv2.THRESH_BINARY)

    mask = refine_saliency_with_grabcut(img,saliency)
    save_image(mask,'hello.jpg')

def save_image(img,dest):
    cv2.imwrite(dest,img)

def create_temp_file_name(id,src_file,dest_dir):
    camera = os.path.basename(src_file).split("_")[1]
    return  os.path.join(dest_dir,str(id)+"_"+camera)


def get_contour_boundary(contours,img):
    # xs = [point[0] for point in contours]
    # ys = [point[1] for point in contours]
    # xs = sorted(xs)
    # ys = sorted(ys)

    x_min,x_max,y_min,y_max = img.shape[0],0,img.shape[1],0

    for points in contours:
        for point in points:
            #print(point)
            x_min = min(x_min,point[0][0])
            x_max = max(x_max, point[0][0])
            y_min = min(y_min, point[0][1])
            y_max = max(y_max, point[0][1])


    ## return lowest x and y
    return x_min,y_min,x_max,y_max



def using_contour(img_rgb):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img = cv2.bilateralFilter(img, 9, 105, 105)
    r, g, b = cv2.split(img)
    equalize1 = cv2.equalizeHist(r)
    equalize2 = cv2.equalizeHist(g)
    equalize3 = cv2.equalizeHist(b)
    equalize = cv2.merge((r, g, b))

    equalize = cv2.cvtColor(equalize, cv2.COLOR_RGB2GRAY)

    ret, thresh_image = cv2.threshold(equalize, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    equalize = cv2.equalizeHist(thresh_image)


    canny_image = cv2.Canny(equalize, 250, 255)
    canny_image = cv2.convertScaleAbs(canny_image)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)

    new, contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    ## remove some contour areas
    contours = contours[:min(10,int(len(contours)*0.8))]

    # c = contours[0]
    # print(len(contours))
    # print(cv2.contourArea(c))
    final = cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
    mask = np.zeros(img_rgb.shape, np.uint8)

    new_image = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    # plt.imshow(new_image, cmap='gray', interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()
    #cv2.waitKey(0)
    # new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # ret, thresh1 = cv2.threshold(new_image_gray, 100, 255, cv2.THRESH_BINARY)
    # final = cv2.bitwise_and(img_rgb, img_rgb, mask=thresh1)
    x_bottom,y_bottom,x_top,y_top = get_contour_boundary(contours,img_rgb)
    return img_rgb[y_bottom:y_top+1,x_bottom:x_top+1]


def cut_image(id,image_file,dest_dir,annotation):
    x_l,y_l,x_t,y_t = get_bounding_box(id,annotation)
    img = crop_image(image_file,x_l,y_l,x_t,y_t)
    new_image = using_contour(img)
    #cv2.imwrite('/home/shibin/Downloads/contour.jpg',new_image)
    #img = reduce_bounding_box(img)
    #test_new_method(img)
    dest_file = create_temp_file_name(id,image_file,dest_dir)
    save_image(new_image,dest_file)

"""
    This functionality is used for checking the file exist or not.
    This is not required in actual running. But for test purpose keeping it.
"""
def file_exist_check(file_name):
    return os.path.exists(file_name)


"""
    Make directory if not exist.
"""
def makedir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        print("Director {} exist".format(dir_name))

def count_set_bits(array,start_x,start_y,dim_x,dim_y,max_overlap):
    count = 0
    limit = dim_x*dim_y-max_overlap
    for i in range(start_x,start_x+dim_x):
        for j in range(start_y,start_y+dim_y):
            if array[i][j][0]==-1: ## only check R
                count+=1
            if count>limit:
                return dim_x*dim_y-count

    return dim_x*dim_y-count

import math
"""
    Return the image paste point.
"""
def find_next_placement(cur_image,merging_img,cur_x,cur_y,max_overlap=2500):
    img_dim_x,img_dim_y = cur_image.shape[0],cur_image.shape[1]
    n,m,rgb = merging_img.shape ## merging image
    max_overlap_one_dir = int(math.sqrt(max_overlap))
    cur_x = max(0,cur_x+np.random.randint(low=-1*max_overlap_one_dir,high=max_overlap_one_dir)) ## create randomness in placement. That is congestion
    cur_y =max(0,cur_y+np.random.randint(low=-1*max_overlap_one_dir,high=max_overlap_one_dir))

    ## scan through the y dimension.

    for i in range(cur_x,img_dim_x):
        for j in range(0,img_dim_y):
            if count_set_bits(cur_image,i,j,n,m,max_overlap)<max_overlap and (i+n)<img_dim_x and (j+m)<img_dim_y:
                return  i,j ## return the start index for merging


    return -1,-1



"""
    Generate Single annotation and image entry
"""
import time
IMG_ID = 1
def get_generated_id():
    global IMG_ID
    IMG_ID += 1
    return IMG_ID

def create_image_metadata(height,width):
    file_name =   str(int(round(time.time() * 1000)))+str(np.random.randint(0,10))+".jpg"
    id = get_generated_id()
    img_metadata = {
        "license": 4,
        "file_name": file_name,
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": height,
        "width": width,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": id
    }
    return img_metadata

def generate_annotaion(id,bbox,category_id):

    annotation ={
        "segmentation": [[204.01, 306.23,206.53, 307.95]],
        "num_keypoints": 15,
        "area": 5463.6864,
        "iscrowd": 0,
        "keypoints": [229, 256, 2, 223, 369, 2],
        "image_id": id,
        "bbox": bbox,
        "category_id": category_id,
        "id": id
    }

    return annotation


"""

    create the random images from sample images
"""
IMG_SHAPE = (1800,1800,3)
def merge_images(n,all_ids,src_dir,dest_dir,annotations,allowed_overlap=2500):

    size = len(all_ids)
    random_index = np.random.randint(0,size,size=n) # generate n items
    np_img_array = np.zeros((IMG_SHAPE))
    np_img_array.fill(-1) # all the value is filled with pure white.
    retain_image_ratio = 0.4 ## 40% chance you retain
    first = True

    start_x,start_y=50,50
    img_metadata = create_image_metadata(IMG_SHAPE[0],IMG_SHAPE[1]) ## global metadata
    generated_annotations = []

    for index in random_index:
        img_id = all_ids[index]
        retain_flip = np.random.random() ## retain the same image
        category_id = annotation.imgToAnns[img_id][0]['category_id']
        if retain_flip>retain_image_ratio or first:
            old_file_name = annotations.imgs[img_id]['file_name']
            src_file_name = create_temp_file_name(img_id,old_file_name,src_dir)## create the cropped file path.
            first = False

        ## extra check for test environment
        if not os.path.exists(src_file_name):
            #print("file is not available {} for merging".format(src_file_name))
            continue ## Pass the ball :)
        else:
            print("merging file{} to {} ".format(src_file_name,img_metadata['file_name']))
        merging_img = cv2.imread(src_file_name)

        t_start_x,t_start_y = find_next_placement(np_img_array,merging_img,start_x,0,allowed_overlap)

        ## break incase of there is no place to merge
        if t_start_x==-1:
            print('no space availbale {} '.format(merging_img.shape))
            break
        start_x, start_y = t_start_x,t_start_y
        print('{},{}.{}.{}'.format(start_x,start_x+merging_img.shape[0],start_y,start_y+merging_img.shape[1]))
        np_img_array[start_x:start_x+merging_img.shape[0],start_y:start_y+merging_img.shape[1]] = merging_img
        bbox = [start_x,merging_img.shape[0],start_y,merging_img.shape[1]]
        generated_annotations.append(generate_annotaion(img_metadata['id'],bbox,category_id))

    ## remove -1 t0 255. This actually take care of the coversion.
    for i in range(IMG_SHAPE[0]):
        for j in range(IMG_SHAPE[0]):
            if np_img_array[i][j][0] ==-1:
                np_img_array[i][j]=[152,159,160]
    return np_img_array.astype('uint8'),img_metadata,generated_annotations



def create_merged_images(n_generate,annotation,src_dir,dest_dir,lowest_cat_n=3,high_cat_n=10):
    max_category = 200


    all_ids = list(annotation.imgs.keys()) ## required for highlevel control
    generated_img_metadatas = []
    generated_annotations = []
    for i in range(n_generate):
        n = np.random.randint(lowest_cat_n,21)
        generated_img, generated_img_metadata,generated_annotation = merge_images(n,all_ids,src_dir,dest_dir,annotation)
        generated_img_metadatas.append(generated_img_metadata)
        generated_annotations.append(generated_annotation)

        ## save generated image file
        generated_location = os.path.join(dest_dir,generated_img_metadata['file_name'])
        cv2.imwrite(generated_location,generated_img)

    ## create all

    final_json = {}
    final_json['images'] = generated_img_metadatas
    final_json['annotations'] = generated_annotations
    final_json['info'] = annotation.dataset['info']
    final_json['licenses'] = annotation.dataset['licenses']
    final_json['categories'] = annotation.dataset['categories']

    ## save the annotation file
    json_location = os.path.join(dest_dir,'annotation.json')
    with open(json_location,'w') as f:
        import json
        json.dump(final_json,f)
















def do_preprocessing_all_image(annotation_file,src_dir,dest_dir):
    annotation = load_annotations(annotation_file)


    for id,ant in annotation.imgs.items():
        print(str(id)+":"+str(ant))
        img_file = ant["file_name"]
        img_path  = os.path.join(src_dir,img_file)
        if os.path.exists(img_path):
            cut_image(id,img_path,dest_dir,annotation)

    # id = 41058
    # print(len(annotation.catToImgs[199]))
    # image_file = '/home/shibin/Downloads/train2019/6954432710621_camera2-6.jpg'
    # for key,value in annotation.imgs.items():
    #     if value['file_name']=='6954432710621_camera2-6.jpg':
    #         print("key={}".format(key))
    # print(annotation.imgs[1])
    # cut_image(id, image_file, '/home/shibin/Downloads/',annotation)


if __name__ == '__main__':
    src_dir = '/home/shibin/Downloads/train2019/'
    dest_dir = '/media/shibin/disk/created'
    annotaion_file = '/home/shibin/Downloads/instances_train2019.json'
    img_dest_dir = '/media/shibin/disk/created_imgs'
    annotation = annotation = load_annotations(annotaion_file)
    #do_preprocessing_all_image('/home/shibin/Downloads/instances_train2019.json',src_dir,dest_dir)
    n_image_to_create = 100
    create_merged_images(n_image_to_create,annotation,dest_dir,img_dest_dir)


