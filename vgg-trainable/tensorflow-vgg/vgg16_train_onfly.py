"""
Simple tester for the vgg16_trainable
"""

#from scipy import misc
import numpy as np
from PIL import Image
import sys

import tensorflow as tf

import vgg16_trainable as vgg16
import utils
import glob
import os
from datetime import datetime

image_dir=sys.argv[1]
annot_dir=sys.argv[2]

num_classes = 2

batch_size = 20

patch_size = 50

n_testing = 20

imgs = []
anns = []
for impath in os.listdir(image_dir):
	im = Image.open(os.path.join(image_dir, impath))
	an = Image.open(os.path.join(annot_dir, impath[0:-4] + '_anno.bmp'))
	imgs.append(im)
	anns.append(an)


def isCancer(ptch):
	w,h,_ = ptch.shape
	count = np.sum(ptch)
	total = w*h
	if count/float(total) >= 0.5:
		return True
	return False

def next_batch(imgs,anns,batch_size,ptch_size,testing=False):
	out_size = 224
	img_batch = np.zeros((batch_size,out_size,out_size,3))
	lbl_batch = np.zeros((batch_size,2))
	for i in range(batch_size):
		if testing:
			idx = np.random.randint(n_testing) + len(imgs)-n_testing
		else:
			idx = np.random.randint(len(imgs)-n_testing)
		im = imgs[idx]
		an = anns[idx]
		w, h = im.size
		x = np.random.randint(w-ptch_size)
		y = np.random.randint(h-ptch_size)
		im_ptch = im.crop((x,   y,   x+ptch_size, y+ptch_size)).resize((out_size, out_size),Image.ANTIALIAS)
		an_ptch = an.crop((x,   y,   x+ptch_size, y+ptch_size))
		img_batch[i,:,:,:] = np.array(im_ptch.getdata()).reshape(im_ptch.size[0], im_ptch.size[1], 3)
		an_arr    = np.array(an_ptch.getdata()).reshape(an_ptch.size[0], an_ptch.size[1], 1)
		if isCancer(an_arr):
			lbl_batch[i,0]=1.0
		else:
			lbl_batch[i,1]=1.0
	return img_batch, lbl_batch


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
		img_batch, lbl_batch = next_batch(imgs, anns, batch_size, patch_size)
		_, l = sess.run([train, cost], feed_dict={images: img_batch, true_out:lbl_batch, train_mode: True})
		if i % 100 == 0:
			print datetime.now(), "step=",i, "loss = ", l
		if i%1000 == 0 and i !=0:
			img_batch, lbl_batch = next_batch(imgs, anns, batch_size, patch_size, testing=True)
			print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={images: img_batch, true_out:lbl_batch, train_mode: False}))
		if i % 1000 == 0 and i !=0 :
			# test save
			vgg.save_npy(sess, './test-save.npy')
