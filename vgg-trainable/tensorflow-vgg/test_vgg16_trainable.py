"""
Simple tester for the vgg16_trainable
"""

#from scipy import misc
import numpy as np
from PIL import Image

import tensorflow as tf

import vgg16_trainable as vgg16
import utils
import glob
import os
from datetime import datetime

image_dir = '/home/bashir/colorectal/train-patch50'
val_dir = '/home/bashir/colorectal/validation-patch50'

num_classes = len(os.listdir(image_dir))

batch_size = 20

max_images = 3000
num_images = 0
for cls in os.listdir(image_dir):
	num_images = num_images + len(os.listdir(os.path.join(image_dir, cls)))
print "number of images: ", num_images
#num_images = max(num_images, max_images)
imgs = np.zeros((num_images, 224, 224, 3))
lbls = np.zeros((num_images, num_classes))
i=0
lbl=0
for cls in os.listdir(image_dir):
	for impath in os.listdir(os.path.join(image_dir, cls)):
		tmpim = misc.imread(os.path.join(image_dir, cls, impath))
		imgs[i,:,:,:] = tmpim.astype('float32')/255.0
		lbls[i,lbl]=1.0
		i=i+1

	lbl=lbl+1


#validation

num_val = 0
for cls in os.listdir(val_dir):
	num_val = num_val + len(os.listdir(os.path.join(val_dir, cls)))
print "number of val images: ", num_val
imgs_val = np.zeros((num_val, 224, 224, 3))
lbls_val = np.zeros((num_val, num_classes))
i=0
lbl=0
for cls in os.listdir(val_dir):
	for impath in os.listdir(os.path.join(val_dir, cls)):
		tmpim = misc.imread(os.path.join(val_dir, cls, impath))
		imgs_val[i,:,:,:] = tmpim.astype('float32')/255.0
		lbls_val[i,lbl]=1.0
		i=i+1
	lbl=lbl+1

#img1 = utils.load_image("./test_data/puzzle.jpeg")
#img1_true_result = [1 if i == 292 else 0 for i in xrange(num_classes)]  # 1-hot result for tiger

#batch1 = img1.reshape((1, 224, 224, 3))


with tf.device('/gpu:0'):
	sess = tf.Session()

	images = tf.placeholder(tf.float32, [None, 224, 224, 3])
	true_out = tf.placeholder(tf.float32, [None, num_classes])
	train_mode = tf.placeholder(tf.bool)

	vgg = vgg16.Vgg16('./vgg16.npy')
	vgg.build(images, train_mode, fine_tuning=True, num_classes=num_classes)

	# print number of variables used: 143667240 variables, i.e. ideal size = 548MB
	#print vgg.get_var_count()

	sess.run(tf.initialize_all_variables())

	# simple 1-step training
	cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
	train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

	correct_prediction = tf.equal(tf.argmax(vgg.prob,1), tf.argmax(true_out,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	for i in xrange(0,10000000):
		batch_idxs = np.random.randint(num_images,size=batch_size)
		img_batch = imgs[batch_idxs, :, :, :]
		lbl_batch = lbls[batch_idxs]
		_, l = sess.run([train, cost], feed_dict={images: img_batch, true_out:lbl_batch, train_mode: True})
		if i % 100 == 0:
			print datetime.now(), "step=",i, "loss = ", l
		if i%1000 == 0 and i !=0:
			print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
	    images: imgs_val[range(0,num_val,num_val/100), :, :, :], true_out: lbls_val[range(0,num_val,num_val/100)], train_mode: False}))
		if i % 1000 == 0 and i !=0 :
			# test save
			vgg.save_npy(sess, './test-save.npy')
