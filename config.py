#!/usr/bin/envpython
#-*-coding:utf-8-*-
#author:YueJietime:2020/8/12
import glob,os

ccae_result_path = 'ccae_result'

save_encode_data = 'encode_result'



model_config={"input_shape":[256,256,1],\
              "embedding":{"is_embedding":False,"embedding_dim":64,"input_dim":256},\
              "structure":{1:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":4,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           2:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":4,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           3:{"is_deconv":False,"is_conv":True,"conv_ker":(30,30),"chanel":8,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           4:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":8,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           5:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":8,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           6:{"is_deconv":False,"is_conv":True,"conv_ker":(15,15),"chanel":4,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           7:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":2,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           8:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":2,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           9:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":2,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                      
                           10:{"is_deconv":True,"is_conv":False,"conv_ker":(3,3),"chanel":4,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           11:{"is_deconv":True,"is_conv":False,"conv_ker":(15,15),"chanel":8,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           12:{"is_deconv":True,"is_conv":False,"conv_ker":(30,30),"chanel":4,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           13:{"is_deconv":True,"is_conv":False,"conv_ker":(3,3),"chanel":1,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                           }
              }

train_config={
        "batch_size":16,
        "save_step":100,
        "check_path":"checkpoint",
        "learning_rate":3e-4,
        "epochs":1000,
        "raw_data":"demo_dataset/crop_1000_erzhihua"
        }


predict_config={"batch_size":1}
