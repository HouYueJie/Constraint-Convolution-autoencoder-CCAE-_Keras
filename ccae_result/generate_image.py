# -l/usr/bin/env python
#-*- coding:utf-8 -*-
# author: YueJie time:2020/9/9
from glob import glob
from matplotlib import pyplot as plt
import pickle
import numpy as np

#read loss data
with open("loss","rb") as f:
    loss =pickle.load(f)
print(loss)

#draw loss pic
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.title('Training loss')
plt.savefig("loss.png")

#read raw image
with open("raw","rb") as f:
    raw =pickle.load(f)
#read reconstruct image
with open("pred","rb") as f:
    pred =pickle.load(f)

raw=np.expand_dims(raw,axis=-1)

#draw raw image
plt.figure(figsize = (20, 4))
print("raw Image1")
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(raw[i, ..., 0], plt.cm.gray)
    plt.title("raw Image_" + str(i))
plt.savefig("raw_image.png")

#draw reconstruct image
plt.figure(figsize = (20, 4))
print("pred Image1")
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(pred[i, ...,0],plt.cm.gray)
    plt.title("pred Image_" + str(i))
plt.savefig("rec_image.png")


