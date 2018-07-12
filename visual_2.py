#-*- coding: UTF-8 -*-  
import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import sys
import pickle
import cv2

caffe_root = '/home/malu/caffe/'  

#deployPrototxt =  caffe_root+'models/bvlc_reference_caffenet/deploy.prototxt'  
#modelFile =  caffe_root+ 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
#meanFile = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
deployPrototxt =  caffe_root+'models/VGGNet/fall/FSSD_300x300/deploy.prototxt'  
modelFile =  caffe_root+ 'models/VGGNet/fall/FSSD_300x300/VGG_fall_FSSD_300x300_iter_15000.caffemodel'
meanFile = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
#imageListFile = '/home/chenjie/DataSet/CompCars/data/train_test_split/classification/test_model431_label_start0.txt'
#imageBasePath = '/home/chenjie/DataSet/CompCars/data/cropped_image'
#resultFile = 'PredictResult.txt'

#网络初始化
def initilize():
    print 'initilize ... '
    sys.path.insert(0, caffe_root + 'python')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net

#取出网络中的params和net.blobs的中的数据
def getNetDetails(image, net):
   

    image = caffe.io.load_image(image)
    # image = caffe.io.resize_image(image,(227,227,1))
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #原来读入的数据形式是H*W*K（0,1,2）,但需要的是K*H*W（2,0,1）
    transformer.set_transpose('data', (2,0,1))
    #计算均值，读数据均值文件
    transformer.set_mean('data', np.load(caffe_root + meanFile ).mean(1).mean(1)) # mean pixel
    #把0-1的数值转化为0-255
    transformer.set_raw_scale('data', 255)  
    # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  
    # the reference model has channels in BGR order instead of RGB
    # set net to batch size of 50
    #将输入图片格式转化成合适格式（与deploy文件相同）
    net.blobs['data'].reshape(1,3,300,300)
    #执行上面设置的图片预处理操作，并将图片载入到blob中
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    #执行测试，前送迭代，即分类
    out = net.forward()

    # for each layer, show the output shape
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)
    
    #网络提取conv1的卷积核
    filters = net.params['conv1_1'][0].data
    with open('FirstLayerFilter.pickle','wb') as f:
       pickle.dump(filters,f)
    vis_square(filters.transpose(0, 2, 3, 1))
    #conv1_1的特征图
    feat = net.blobs['conv1_1'].data[0, :64]
    #with open('FirstLayerOutput.pickle','wb') as f:
    #   pickle.dump(feat,f)
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv1_1.png")
    plt.show()
    #conv1_2的特征图
    feat = net.blobs['conv1_2'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv1_2.png")
    plt.show()
    #pool1的特征图
    pool = net.blobs['pool1'].data[0,:64]
    #with open('pool1.pickle','wb') as f:
    #   pickle.dump(pool,f)
    vis_square(pool,padval=1)
    plt.savefig(caffe_root+"images2/pool1.png")
    plt.show()

    #conv2_1的特征图
    feat = net.blobs['conv2_1'].data[0, :64]
    #with open('FirstLayerOutput.pickle','wb') as f:
    #   pickle.dump(feat,f)
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv2_1.png")
    plt.show()

    #conv2_2的特征图
    feat = net.blobs['conv2_2'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv2_2.png")
    plt.show()
    #pool2的特征图
    pool = net.blobs['pool2'].data[0,:64]
    #with open('pool1.pickle','wb') as f:
    #   pickle.dump(pool,f)
    vis_square(pool,padval=1)
    plt.savefig(caffe_root+"images2/pool2.png")
    plt.show()

    #conv3_1的特征图
    feat = net.blobs['conv3_1'].data[0, :64]
    #with open('FirstLayerOutput.pickle','wb') as f:
    #   pickle.dump(feat,f)
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv3_1.png")
    plt.show()
    #conv3_2的特征图
    feat = net.blobs['conv3_2'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv3_2.png")
    plt.show()
    #conv3_3的特征图
    feat = net.blobs['conv3_3'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv3_3.png")
    plt.show()
    #pool3的特征图
    pool = net.blobs['pool3'].data[0,:64]
    #with open('pool1.pickle','wb') as f:
    #   pickle.dump(pool,f)
    vis_square(pool,padval=1)
    plt.savefig(caffe_root+"images2/pool3.png")
    plt.show()
    
    #conv4_1的特征图
    feat = net.blobs['conv4_1'].data[0, :64]
    #with open('FirstLayerOutput.pickle','wb') as f:
    #   pickle.dump(feat,f)
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv4_1.png")
    plt.show()
    #conv4_2的特征图
    feat = net.blobs['conv4_2'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv4_2.png")
    plt.show()
    #conv4_3的特征图
    feat = net.blobs['conv4_3'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv4_3.png")
    plt.show()
    #pool4的特征图
    pool = net.blobs['pool4'].data[0,:64]
    #with open('pool1.pickle','wb') as f:
    #   pickle.dump(pool,f)
    vis_square(pool,padval=1)
    plt.savefig(caffe_root+"images2/pool4.png")
    plt.show()

    #conv5_1的特征图
    feat = net.blobs['conv5_1'].data[0, :64]
    #with open('FirstLayerOutput.pickle','wb') as f:
    #   pickle.dump(feat,f)
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv5_1.png")
    plt.show()
    #conv5_2的特征图
    feat = net.blobs['conv5_2'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv5_2.png")
    plt.show()
    #conv5_3的特征图
    feat = net.blobs['conv5_3'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv5_3.png")
    plt.show()
    #pool5的特征图
    pool = net.blobs['pool5'].data[0,:64]
    #with open('pool1.pickle','wb') as f:
    #   pickle.dump(pool,f)
    vis_square(pool,padval=1)
    plt.savefig(caffe_root+"images2/pool5.png")
    plt.show()
   
    
    #conv6_1的特征图
    feat = net.blobs['conv6_1'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv6_1.png")
    plt.show()
    #conv6_2的特征图
    feat = net.blobs['conv6_2'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv6_2.png")
    plt.show()
     
    #conv7_1的特征图
    feat = net.blobs['conv7_1'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv7_1.png")
    plt.show()
    #conv7_2的特征图
    feat = net.blobs['conv7_2'].data[0, :64]
    vis_square(feat,padval=1)
    plt.savefig(caffe_root+"images2/conv7_2.png")
    plt.show()
    
    #conv1_1输出后的官方直方图
    #feat=net.blobs['conv1_1'].data[0]
    #plt.subplot(2,1,1)
    #plt.plot(feat.flat)
    #plt.subplot(2,1,2)
    #_=plt.hist(feat.flat[feat.flat>0],bins=100)
    #plt.show()


# 此处将卷积图和进行显示，
def vis_square(data, padsize=1, padval=0 ):#padsize为特征图间距 padval用于调整亮度
    #归一化
    data -= data.min()
    data /= data.max()
    
    #让合成图为正方形，根据data中图片数量data.shape,计算最后输出时每行每列图片数n
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    #合并卷积图到一个图像中
    #先将padding后夫人data分成n*n张图像
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    #再将（n,W,n,H）变换成（n*W，n*H）
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data.shape
    plt.imshow(data)
    
    

if __name__ == "__main__":
    net = initilize()
    testimage = '/home/malu/caffe/examples/images/1414.jpg'
    getNetDetails(testimage, net)
