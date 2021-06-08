#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: YueJie time:2020/8/12

import glob
from keras.models import Model, load_model
import config as C
import model as M
import utils as U
import pickle
import sys
import numpy as np
import cv2 as cv
import os
from keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"]='-1'

def save(data,file_name):
    with open(file_name,"wb")as f:
        pickle.dump(data,f)


if __name__=="__main__":
    check_path = C.train_config["check_path"]
    if sys.argv[1]=="train":
        if sys.argv[2]=='gpu':
            os.environ["CUDA_VISIBLE_DEVICES"]='1'
        elif sys.argv[2]=='cpu':
            os.environ["CUDA_VISIBLE_DEVICES"]='-1'
        else:
            print('Please choose a model type (gpu/cpu)')
        if glob.glob(check_path+'/*')!=[]:
            new_train=input("是否删除旧的权重？(yes or no)")
        else:
            new_train="no"
        while new_train not in ["yes","no"]:
            new_train=input("是否删除旧的权重？(yes or no)")
        if new_train=="yes" and glob.glob(check_path+'/*')!=[]:
            os.system("rm %s/*"%check_path+"/*")


        CAE=M.CAE(C.model_config,C.train_config)
        
        if glob.glob(check_path+'/*')!=[]:
            print("已经存在CAE模型，继续训练！")
            path = sorted(glob.glob(check_path+'/*'),key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print("已经存在CAE模型%s，继续训练！"%(path.split("/")[-1]))
            CAE.cae.load_weights(path)
        else:
            print("没有训练CAE模型，重新训练！")

        
        raw_data_path_lst = glob.glob(C.train_config["raw_data"]+"/*")
        train_batch_data_generator=U.generate_arrays_from_file(raw_data_path_lst,C.train_config["batch_size"],0)
        
        checkpointer = ModelCheckpoint(os.path.join(check_path,"model_{epoch:03d}.h5"),\
                                       verbose=1,save_weights_only=False,period=C.train_config["save_step"])
        CAE_train = CAE.cae.fit(train_batch_data_generator,epochs=C.train_config["epochs"],\
                                steps_per_epoch=int(len(raw_data_path_lst)/C.train_config["batch_size"]),callbacks=[checkpointer],shuffle=True)
        
        #Loss and prediction
        loss = CAE_train.history['loss']
        save(loss,C.ccae_result_path+"/loss")
        img=next(train_batch_data_generator)
        raw_test=img[0]
        pred = CAE.cae.predict(raw_test)
        save(raw,C.ccae_result_path+"/raw")
        save(pred,C.ccae_result_path+"/pred")

    #extract encode data
    if sys.argv[1]=="encode": 
        if sys.argv[2]=='gpu':
            os.environ["CUDA_VISIBLE_DEVICES"]='1'
        elif sys.argv[2]=='cpu':
            os.environ["CUDA_VISIBLE_DEVICES"]='-1'
        else:
            print('Please choose a model type (gpu/cpu)')
        CAE=M.CAE(C.model_config,C.train_config)
        
        if glob.glob(check_path+'/*')!=[]:
            path = sorted(glob.glob(check_path+'/*'),key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print("已经存在CAE模型%s，可以进行预测！"%(path.split("/")[-1]))
            CAE.cae.load_weights(path)
        else:
            print("没有训练CAE模型，需要重新训练！")
            exit()

        raw_data_path_lst = glob.glob(C.train_config["raw_data"]+"/*")
        encode_batch_data_generator=U.generate_arrays_from_file_2(raw_data_path_lst,C.predict_config["batch_size"],0)
        total_data=len(raw_data_path_lst)


        layer_name = input("name_middle:   ")
        Model_MiddleLayer = Model(inputs = CAE.input_img,outputs = Model.get_layer(CAE.cae,layer_name).output)
        
        with open(C.save_encode_data,'w')as f:
            for n in range(0,int(total_data/C.predict_config["batch_size"])):
                file_,input_,output_=next(encode_batch_data_generator)
                MiddleLayer_output = Model_MiddleLayer.predict(input_)
                for ff,data in zip(file_,MiddleLayer_output):
                    data=np.reshape(data,[-1,])
                    f.write("%s\t%s\n"%(ff,"<=>".join([str(x)for x in list(data)])))
