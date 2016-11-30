#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn16_vgg
import utils
from tensorflow.contrib.session_bundle import exporter

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops
img1 = skimage.io.imread("./test_data/testB_5.bmp")
images = tf.placeholder("float")
feed_dict = {images: img1}
batch_images = tf.expand_dims(images,0)
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
vgg_fcn = fcn16_vgg.FCN16VGG()
with tf.name_scope("content_vgg"):
    vgg_fcn.build(batch_images, debug=True)




saver = tf.train.Saver(sharded=True)
with tf.Session() as sess:

    saver.restore(sess, "./train_fcn16_36k/fcn-36000")
    #vgg_fcn = fcn8_vgg.FCN8VGG()
    #with tf.name_scope("content_vgg"):
        #vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")

    #init = tf.initialize_all_variables()
    #sess.run(tf.initialize_all_variables())

    print('Running the Network')
    #tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    #down, up = sess.run(tensors, feed_dict=feed_dict)

    #down_color = utils.color_image(down[0])
    #up_color = utils.color_image(up[0])

    export_path = sys.argv[-1]
    for var in tf.get_collection(tf.GraphKeys.VARIABLES):
	print var.name
    print 'Exporting trained model to', export_path
    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
	graph_def=None,
    	#sess.graph.as_graph_def(),
    	named_graph_signatures={
        	'inputs': exporter.generic_signature({'images': images}),
        	'outputs': exporter.generic_signature({'segment': vgg_fcn.pred_up})})
    model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)
