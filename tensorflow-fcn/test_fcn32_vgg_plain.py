#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf

import fcn32_vgg
import utils

from tensorflow.python.framework import ops

def softmax(target, axis, name=None):
  with tf.op_scope([target], name, 'softmax'):
    max_axis = tf.reduce_max(target, axis, keep_dims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
    softmax = target_exp / normalize
    return softmax

os.environ['CUDA_VISIBLE_DEVICES'] = ''

#img1 = skimage.io.imread("./test_data/intestinalparasites-0001.jpg")
img1 = skimage.io.imread("./test_data/plasmodium-2525.jpg")

with tf.Session() as sess:
	
	images = tf.placeholder("float")
	feed_dict = {images: img1}
	batch_images = tf.expand_dims(images, 0)

	vgg_fcn = fcn32_vgg.FCN32VGG()
	with tf.name_scope("content_vgg"):
		vgg_fcn.build(batch_images, num_classes=2,debug=True)

	print('Finished building Network.')

	init = tf.initialize_all_variables()
	sess.run(tf.initialize_all_variables())

	print('Running the Network')

	
	#y_conv=softmax(vgg_fcn.upscore, axis =1)

	#self.pred_up = tf.argmax(self.upscore, dimension=3)
	

	up = sess.run(vgg_fcn.upscore, feed_dict=feed_dict)
	print up
	
	
	softm = sess.run(softmax(up, axis =3))
	
 	
	
	up = softm[:,:,:,1]
	filtered =  np.where(up>0.5,1,0)
	#filtered =  np.argmax(np.where(softm>n,n,0),3)
	up = filtered	
	up_color = utils.color_image(up[0])
	
	scp.misc.imsave('test_intes_fcn32_upsampled.png', up_color)


