import numpy as np
from numpy import dot
import scipy.linalg as la
# import cv2
from matplotlib import pyplot as plt

import cPickle    
import os    
import json  
import pylab
from PIL import Image

import random

k=0
for line in open("flower_data/jpg/files.txt"):
    k=k+1

print k

#initialize
line_file=0
label =0
olivettifaces=np.zeros((k,28*28))
olivettifaces_label=np.zeros(k)
i=0
j=1
label_check=0

#for i in xrange(k):
for line in open("flower_data/jpg/files.txt"):
    print i+1
    line=line.strip('\n')
    line_file="flower_data/jpg/"+line
    # print line_4
    imgage = Image.open(line_file)
    imgage = imgage.resize((300,900))
    img2 = np.asarray(imgage, dtype='float32')/256
    img2=img2.flat
    img3=np.array(img2)
    img3.resize((28,28))
    img_deal=np.array(img3)
    print img_deal.shape
    img_what=np.ndarray.flatten(img_deal)
    olivettifaces[i]=img_what
    olivettifaces_label[i]=label_check
    j=j+1
    if j>80:
    	label_check=label_check+1
    	j=1   
    print olivettifaces_label[i]
    i=i+1

print i

print olivettifaces.shape
print olivettifaces_label.shape

indices = range(k) # indices = the number of images in the source data set  
random.shuffle(indices)  
key=0
last_image = olivettifaces
last_label = olivettifaces_label
for i in indices:  
    last_image[key] = olivettifaces[i]
    last_label[key] = olivettifaces_label[i]  
    key=key+1

print last_image.shape
print last_label.shape

write_file=open('flower.pkl','wb')
cPickle.dump([[last_image[0:int(k/2)],last_label[0:int(k/2)]],
              [last_image[int(k/2)+1:3*int(k/4)],last_label[int(k/2)+1:3*int(k/4)]],
              [last_image[3*int(k/4)+1:int(k)],last_label[3*int(k/4)+1:int(k)]]],write_file,-1)  

print last_label[int(k/2)]
print last_label[int(k/2)+1]
print last_label[3*int(k/4)]
print last_label[3*int(k/4)+1]
print last_label[int(k)-1]
write_file.close()
