from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf


def get_deconv_filter(f_shape, stddev):
    width = f_shape[0]
    heigh = f_shape[1]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    #bilinear[f,f] = 1.0
    weights = 0.01*stddev*np.random.randn(f_shape[0], f_shape[1], f_shape[2], f_shape[3])
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    print(weights)
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def _upscore_layer(bottom, shape,
                   num_classes, name, debug,
                   ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.pack(new_shape)

        logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        f_shape = [ksize, ksize, num_classes, in_features]

        # create
        num_input = ksize * ksize * in_features / stride
        print("num_input: ", num_input)
        stddev = (2 / num_input)**0.5
        #stddev = 0.0

        weights = get_deconv_filter(f_shape, stddev)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv)],
                              message='Shape of %s' % name,
                              summarize=4, first_n=1)

    return deconv

a = np.zeros((1,4,4,2),dtype=np.float32)

a[0,2,2,1] = 1.0
a[0,1,1,0] = 2.0

inp = tf.placeholder(shape=[1,4,4,2],dtype=tf.float32)
#btm = tf.placeholder(shape=[1,8,8,2],dtype=tf.float32)
outp = _upscore_layer(inp, shape=None, num_classes=2, name='test', debug=True)

print(a)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    b = sess.run(outp, feed_dict={inp:a})
    print(b)
