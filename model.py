#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: YueJie time:2020/8/12

import keras
from keras.layers import  Input, Add,Conv2D, Embedding, Reshape, Flatten, Dense, MaxPooling2D, UpSampling2D,Conv2DTranspose
from keras.models import Model
from keras import optimizers
import config as CF
import os
from keras import backend as K
import tensorflow as tf

class CAE():
    def __init__(self,model_config,train_config):
        self.conf=model_config
        self.train_conf=train_config
        self.build_input()
        self.encode()
        self.decode()
        self.build()

    def build_input(self):
        self.input_img=Input(shape=self.conf["input_shape"])#,dtype=K.floatx())
        #Embedding
        if self.conf["embedding"]["is_embedding"]:
            self.embed_dim=self.conf["embedding"]["embedding_dim"]
            self.input_dim=self.conf["embedding"]["input_dim"]
            tmp=Flatten()(self.input_img)    
            tmp = Embedding(self.input_dim,self.embed_dim,embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(tmp)
            resize_shape=self.conf["input_shape"][:2]+[self.embed_dim]
            self.flow = Reshape(resize_shape)(tmp)
            
        #no_embedding
        else:
            if len(self.conf["input_shape"])<3:
                resize_shape=self.conf["input_shape"]+[1]
                self.flow=Reshape(resize_shape)(self.input_img)
            else:
                self.flow=self.input_img
    
    #encoder_layer
    def build_one_layer(self,stru):
        #Convolution layer
        if stru["is_conv"]:
            try:
                self.flow = Conv2D(stru["chanel"], stru["conv_ker"],strides=stru["conv_stride"],activation="relu", padding=stru["padding"])(self.flow)
            except Exception as e:
                print(e)
                print(stru)
                print("Please correctly configure convolutional parameters")
                exit()

        #Pooling layer
        if stru["is_pool"]:
            try:
                self.flow=MaxPooling2D(pool_size = stru["pool_ker"],strides=stru["pool_stride"],padding='same')(self.flow)
            except Exception as e:
                print(e)
                print(stru)
                print("请配置池化参数")
                exit()
        
        #Connection layer
        if stru["is_full"]: 
            try:
                #Flatten layer
                if stru["need_flat"]:
                    self.middle_shape=self.flow.get_shape().as_list()[1:] 
                    print("The middle data size:",self.middle_shape)
                    self.flow = Flatten()(self.flow)
                #Dense layer
                self.flow = Dense(stru["neu_num"],activation=stru["activate"])(self.flow)
            except Exception as e:
                print(e)
                print(stru)
                print("Please correctly configure convolutional parameters")
                exit()
    
    #Decoder layer
    def build_one_layer_decode(self,stru,layer):
        #Connection layer
        if stru["is_full"]: 
            if stru["need_flat"]:
                number_pre=1
                for shape in self.middle_shape:
                    number_pre *=shape
                self.flow = Dense(number_pre,activation=stru["activate"])(self.flow)
                if layer == 1:
                    pass
                else:
                    self.flow = Reshape(self.middle_shape)(self.flow)
            else:
                self.flow=Dense(stru["neu_num"],activation=stru["activate"])(self.flow)
        #UpSampling layer
        if stru["is_pool"]:  
             self.flow = UpSampling2D(size=stru["pool_stride"])(self.flow)
        
        #Deconvolution layer
        if stru["is_deconv"]:  
             self.flow = Conv2DTranspose(stru["chanel"], stru["conv_ker"], strides=stru["conv_stride"],activation="relu",padding='same')(self.flow)

    def get_shape(self,data):
        shapes=data.get_shape().as_list()
        dim=1
        for i in shapes[1:]:
            dim *=i
        return dim
    
    #Building encoder layer 
    def encode(self):
        structure=self.conf["structure"]
        for layer,content in sorted(structure.items(),key=lambda x:x[0]):
            if content["is_conv"]==True:
                self.build_one_layer(content)
        self.encode_result=self.flow
        
        #Obtain the encoded data l2 constraint
        dim=self.get_shape(self.encode_result)
        self.encode_l2_=tf.reshape(self.encode_result,[-1,dim])
        self.encode_l2 = K.mean(K.square(self.encode_l2_),axis=-1)         
    
    #Building decoder layer
    def decode(self):
        structure=self.conf["structure"]
        for layer,content in sorted(structure.items(),key=lambda x:x[0]):
            if content["is_deconv"]==True:
                print(layer,content)
                if layer==len(structure) and content["is_full"]:continue
                self.build_one_layer_decode(content,layer)
        self.out_put=self.flow

    #Build Model
    def build(self):
        self.cae = Model(self.input_img,self.out_put)
        
        #builde loss
        self.pix_err=(K.square(self.input_img-self.out_put))
        dim=self.get_shape(self.pix_err)
        self.pix_err=tf.reshape(self.pix_err,[-1,dim])
        self.loss_pixels=K.mean(self.pix_err,axis=-1)
        self.loss_final=self.loss_pixels+self.encode_l2
        self.cae.add_loss(self.loss_final)
        
        #budile optimizer
        self.adam = optimizers.Adam(lr=self.train_conf["learning_rate"], beta_1=0.9, beta_2=0.999,epsilon=None, decay= 0.0 , amsgrad=False)
    
        self.cae.compile(optimizer = self.adam)
        #obtain model structure and parameters
        self.cae.summary()


if __name__=="__main__":
    train_config={"batch_size":8,
                  "learning_rate":0.001,
                  "epochs":25,#数据训练循环周期
                  "raw_data":"./zhiwen_gaoqing_data",
                                    }
    model_config={"input_shape":[64,64,144],\
                  "embedding":{"is_embedding":False,"embedding_dim":64,"input_dim":256},\
                  "structure":{1:{"is_deconv":False,"is_conv":True,"conv_ker":(9,9),"chanel":64,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               2:{"is_deconv":False,"is_conv":True,"conv_ker":(9,9),"chanel":64,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               3:{"is_deconv":False,"is_conv":True,"conv_ker":(9,9),"chanel":64,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               4:{"is_deconv":False,"is_conv":True,"conv_ker":(7,7),"chanel":64,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               5:{"is_deconv":False,"is_conv":True,"conv_ker":(7,7),"chanel":64,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               6:{"is_deconv":False,"is_conv":True,"conv_ker":(7,7),"chanel":64,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               7:{"is_deconv":False,"is_conv":True,"conv_ker":(5,5),"chanel":64,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               8:{"is_deconv":False,"is_conv":True,"conv_ker":(5,5),"chanel":64,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               9:{"is_deconv":False,"is_conv":True,"conv_ker":(5,5),"chanel":64,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               10:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":64,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               11:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":64,"conv_stride":(1,1),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                               12:{"is_deconv":False,"is_conv":True,"conv_ker":(3,3),"chanel":64,"conv_stride":(2,2),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\

                               13:{"is_deconv":True,"is_conv":False,"conv_ker":(3,3),"chanel":144,"conv_stride":(16,16),"padding":"same","is_pool":False,"pool_ker":(3,3),"pool_stride":(2,2),"is_full":False},\
                              }
                 }
    C1=CAE(model_config,train_config)
    print(C1.encode_result)
