#coding=utf-8  

#各层权值参数可视化
""" 
Caffe visualize 
Display and Visualization Functions and caffe model. 

Copyright (c) 2017 Matterport, Inc. 
Licensed under the MIT License (see LICENSE for details) 
Written by wishchin yang 

MayBe I should use jupyter notebook 
"""  

import numpy as np  
import matplotlib.pyplot as plt  
import os,sys,caffe  
#%matplotlib inline  


#编写一个函数，用于显示各层的参数  
def show_feature(data, padsize=1, padval=0):  
    data -= data.min();  
    data /= data.max();  
	  
   # force the number of filters to be square  
    n = int(np.ceil(np.sqrt(data.shape[0])));  
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3);  
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval));  
  
   # tile the filters into an image  
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)));  
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:]);  
    plt.imshow(data);  
    plt.axis('off');  
    plt.show()


#第一个实例，测试cifar10模型  
def mainex():  
  
    caffe_root='/home/malu/caffe/';  
    os.chdir(caffe_root);  
    sys.path.insert(0,caffe_root+'python');  
  
    plt.rcParams['figure.figsize'] = (10, 10);  
    plt.rcParams['image.interpolation'] = 'nearest';  
    plt.rcParams['image.cmap'] = 'gray';  
  
    net = caffe.Net(caffe_root + 'models/VGGNet/fall/FSSD_300x300/deploy.prototxt',  
          caffe_root + 'models/VGGNet/fall/FSSD_300x300/VGG_fall_FSSD_300x300_iter_15000.caffemodel',  
          caffe.TEST);  
              
    [(k, v[0].data.shape) for k, v in net.params.items()];  
  
    # 第一个卷积层，参数规模为(32,3,5,5)，即32个5*5的3通道filter  
    weight = net.params["conv1_1"][0].data  #参数有两中类型：参数有两种类型：权值参数和偏置项。分别用params["conv1"][0] 和params["conv1"][1] 表示 。我们只显示权值参数，因此用params["conv1"][0] 
    print(weight.shape);  
    show_feature(weight.transpose(0, 2, 3, 1));  
  
    # 第二个卷积层的权值参数，共有32*32个filter,每个filter大小为5*5  
    weight = net.params["conv2_1"][0].data;  
    print weight.shape;  
    show_feature( weight.reshape(128*64, 3, 3)[:256]);  
  
    # 第三个卷积层的权值，共有64*32个filter,每个filter大小为5*5，取其前1024个进行可视化  
    weight = net.params["conv3_1"][0].data ;  
    print weight.shape ;  
    show_feature(weight.reshape(256*128, 3, 3)[:256]);  

if __name__ == '__main__':  
    import argparse  
    mainex();  

