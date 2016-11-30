"""
Simple tester for the vgg16_trainable
"""

from scipy import misc
import numpy as np
from PIL import Image

import tensorflow as tf

import vgg16_trainable as vgg16
import utils
import glob
import os
from datetime import datetime

val_dir = '/media/bashir/10e69c97-5ae7-4ec6-890d-6ab57ba4f1bd/bashir/Plosmodium/validation224-1000'
#val_dir = '/media/bashir/10e69c97-5ae7-4ec6-890d-6ab57ba4f1bd/bashir/Plosmodium/validation_intestinal'
#val_dir = '/media/bashir/10e69c97-5ae7-4ec6-890d-6ab57ba4f1bd/bashir/Plosmodium/intestinal1000'
train_dir = './train'
num_classes = len(os.listdir(val_dir))


#validation

num_val = 0
for cls in os.listdir(val_dir):
	num_val = num_val + len(os.listdir(os.path.join(val_dir, cls)))

num_val = min(num_val, 1000)

print "number of val images: ", num_val
imgs_val = np.zeros((num_val, 224, 224, 3))
lbls_val = np.zeros((num_val, num_classes))
i=0
lbl=0

count =0
for cls in os.listdir(val_dir):
	for impath in os.listdir(os.path.join(val_dir, cls)):
		#if count == num_val/2:
		#	count = 0
		#	print('next class')
		#	break
		#print os.path.join(val_dir, cls, impath)	 	
		tmpim = misc.imread(os.path.join(val_dir, cls, impath))
				
		imgs_val[i,:,:,:] = tmpim.astype('float32')/255.0
		lbls_val[i,lbl]=1.0
		i=i+1

		count+=1
		
	lbl=lbl+1

print "num_val", num_val
print lbls_val
#img1 = utils.load_image("./test_data/puzzle.jpeg")
#img1_true_result = [1 if i == 292 else 0 for i in xrange(num_classes)]  # 1-hot result for tiger

#batch1 = img1.reshape((1, 224, 224, 3))


with tf.Graph().as_default():
	sess = tf.Session()

	images = tf.placeholder(tf.float32, [None, 224, 224, 3])
	true_out = tf.placeholder(tf.float32, [None, num_classes])
	train_mode = tf.placeholder(tf.bool)

	vgg = vgg16.Vgg16('./vgg16.npy')
	vgg.build(images, train_mode, fine_tuning=True, num_classes=num_classes)
	
	# print number of variables used: 143667240 variables, i.e. ideal size = 548MB
	#print vgg.get_var_count()
	
	sess.run(tf.initialize_all_variables())
	#
	# simple 1-step training
	cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
	train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

	correct_prediction = tf.equal(tf.argmax(vgg.prob,1), tf.argmax(true_out,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver(tf.all_variables(), max_to_keep = 2)
	path = os.path.join(train_dir, 'combo')
	saver.restore(sess, path)
	
	sum_ = 0
	for i in xrange(0,1000,100):
		sum_+=accuracy.eval(session=sess, feed_dict={
	    images: imgs_val[range(i,i+100), :, :, :], true_out: lbls_val[range(i,i+100)], train_mode: False})
		print sum_
	
	print("test accuracy"+str(sum_/10))
			
