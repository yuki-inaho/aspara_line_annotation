import pandas as pd
import cv2
import numpy as np
import os
import glob
import pdb

img_files = glob.glob("./data/img/*")
file = img_files[0]
annotation_files = glob.glob("./data/annotation/*")
ann = annotation_files[0]

jsn_file = pd.read_json(ann)
table = jsn_file.transpose().reset_index()[["filename","regions"]]

a = 0
for row in range(table.shape[0]):
    data = table["regions"][row]
    len_data = len(data)

    #anno_img = np.zeros([256,256],dtype=np.uint8)
    _img = cv2.imread(file)
    ann_img = np.zeros(_img.shape)

    for i in range(len_data):
#    for i in range(1):    
        label = table["regions"][row][i]["region_attributes"]["label"]
        shattr = table["regions"][row][i]["shape_attributes"]
        ptr_x = shattr["all_points_x"]
        ptr_y = shattr["all_points_y"]
        
        points_poly = np.array(list(zip(ptr_x,ptr_y)))

                #pseudo_img = cv2.fillPoly(pseudo_img,pts = [points_poly],color =(255,255,255))
        
        if label == 'pole':
            color = (255,0,0)
        if label == 'aspara_harvest':
            color = (0,255,0)
        if label == 'aspara_parent':
            color = (0,0,255)
        ann_img = cv2.polylines(ann_img,[points_poly],True,color)
        
    cv2.imwrite("./data/label/%03d.png" %(a), ann_img)
    a += 1
