# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:22:18 2020

@author: Jacky0628
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(r'E:\ACV\MangoClassify\C1-P1_Train\00002.jpg')
img = img[:,:,::-1]
img2 = np.power(img/float(np.max(img)), 1/1.5)
#img2 = cv2.equalizeHist(img)
res = np.hstack((img,img2)) #stacking images side-by-side
plt.figure()
plt.imshow(res)
plt.show()