# -*- coding: utf-8 -*-
from os import listdir
import os
import multiprocessing
from copy import deepcopy

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt  
import skimage as sk
import skimage.io as io
from skimage import data as da
from skimage.segmentation import mark_boundaries
import cv2

import cProfile as profile
runt = profile.run

from tools import mapp,show,getPoltName,performance,normalizing,random,loga
from tools import saveData,loadData

from warnings import filterwarnings
filterwarnings('ignore')



a=random(3,5)
G = {} # G is a global var to save value for DEBUG in funcation





IMG_DIR = r'test/'
COARSE_DIR ='test/'


from boxx import addPathToSys

addPathToSys(__file__, '../auto_synthesis')
#from config import cropedDir


LABEL_DATA_DIR = os.path.dirname(IMG_DIR)+'/LabelData/'
if not os.path.isdir(LABEL_DATA_DIR):
    os.mkdir(LABEL_DATA_DIR)
    
IMG_NAME_LIST = filter(lambda x:x[-3:]=='jpg',listdir(IMG_DIR))
#allMethods = list(set(map(lambda x: x[x.rindex('_')+1:-4],listdir(COARSE_DIR))))

if __name__ == '__main__':
    from algorithm import readImg,getCoarseDic,showpr
    import algorithm as alg  # main algorithm
    from algorithm import buildImgs,mergeImgs
    
    from analysis import plotMethods

    
    IMG_NAME_LIST=IMG_NAME_LIST[::]
    # build MY* 创造
    coarseMethods=['DISC2','QCUT']
    coarseMethods=[]
    
    buildMethods=['MY1','MY4']
    num = len(IMG_NAME_LIST)
#    num = 0
    for name in IMG_NAME_LIST[:num]:
        buildImgs(name,buildMethods,coarseMethods)
    
    
    # merge methods to make ME* 融合
    mergeMethods=['MY4','DISC2']
    for name in IMG_NAME_LIST[:num]:

        mergeImgs(name,mergeMethods)


    showMethods = ["MY4","ME1","FT","GC","RC", "DRFI", "GMR","QCUT","DISC2"]
    showMethods += filter(lambda x:x not in showMethods,buildMethods)
    
    data = plotMethods(showMethods,
                       num=num,
                       save=getPoltName(coarseMethods,IMG_DIR)
                       )












