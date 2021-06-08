#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: YueJie time:2020/8/12

import numpy as np
from PIL import Image as I
import glob
import config as C
import random


def read_image(batch_img_path_lst,show=False):
    images = []
    for img_path in batch_img_path_lst:
        img = I.open(img_path).convert("L").resize((256,256),I.ANTIALIAS)
        img=np.array(img)
        images.append(img)
    images_arr=np.array(images)
    
    if show:
        print("The size of image is: %s"%images_arr.shape)
    return images_arr


def transform(n,L):
    if n<L:
        return n
    else:
        ret=n-int(n/L)*L
        return ret  

def generate_arrays_from_file(img_path_lst,batch_size,which_seg):
    print("Start generating batch data for model training")
    L=int(len(img_path_lst)/batch_size)
    while True:
        which_seg=transform(which_seg,L)
        this_path_lst=img_path_lst[which_seg*batch_size:(which_seg+1)*batch_size]
        this_images=read_image(this_path_lst,show=False)    
        which_seg +=1
    
        yield this_images,this_images


def generate_arrays_from_file_2(img_path_lst,batch_size,which_seg):
    print("Start generating batch data for model encoding data")
    L=int(len(img_path_lst)/batch_size)
    while True:
        which_seg=transform(which_seg,L)
        this_path_lst=img_path_lst[which_seg*batch_size:(which_seg+1)*batch_size]
        this_images=read_image(this_path_lst,show=False)
        which_seg +=1
        
        yield this_path_lst,this_images,this_images

'''
def generate_arrays_from_256_data(middle_data_lst,batch_size,which_seg):
    L=int(len(middle_data_lst)/batch_size)
    print(L)
    while True:
        which_seg=transform(which_seg,L)
        this_path_lst=middle_data_lst[which_seg*batch_size:(which_seg+1)*batch_size]
        this_images=read_image(this_path_lst,show=False)
        which_seg +=1
        #print("count:"+str(which_seg))
        yield this_images,this_images

#提取压缩数据
def generate_arrays_from_middle_data2(middle_img_lst,batch_size,which_seg):
    print("生成batch数据")
    L=int(len(middle_img_lst)/batch_size)
    print(L)
    while True:
        which_seg=transform(which_seg,L)
        this_path_lst=middle_img_lst[which_seg*batch_size:(which_seg+1)*batch_size]
        this_images=read_image(this_path_lst,show=False)
        which_seg +=1
        #print("count:"+str(which_seg))
        yield this_path_lst,this_images,this_images
'''

if __name__=="__main__":
    raw_data_path_lst = glob.glob(C.train_config["raw_data"]+"/*")
    batch_data_generator=generate_arrays_from_file(raw_data_path_lst,C.train_config["batch_size"],0)
    a=next(batch_data_generator)
    a=next(batch_data_generator)
    print("a",a)
    print("lena",len(a))
    print("lena0",len(a[0]))
    print("lena00",len(a[0][0]))
    print("lena000",len(a[0][0][0]))


