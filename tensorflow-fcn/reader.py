#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 qingze <qingze@localhost.localdomain>
#
# Distributed under terms of the MIT license.

"""

"""
# import codecs, os
# import numpy as np
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf


class Reader(object):
    def __init__(self, data_dir, batch_size = 64, num_threads = 16, num_epochs = None,
                     crop_size = None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.num_epochs = num_epochs
        self.filename_queue = []

    def read_file_list(self, path):
        images = []
        labels = []
        for line in open(path):
            images.append(os.path.join(self.data_dir,'Cell_JPEGImages-small', line.strip() + '.jpg'))
            labels.append(os.path.join(self.data_dir, 'Cell_SegmentationClass-small', line.strip() + '.png'))

        return tf.pack(images), tf.pack(labels)

    def read_and_decode(self):
        image_name = tf.read_file(self.filename_queue[0])
        image = tf.image.decode_jpeg(image_name, channels = 3)
        #image = tf.image.resize_images(image, [320, 480])
        image = tf.image.resize_images(image, [522, 775])
        image /= 255.

        label_name = tf.read_file(self.filename_queue[1])
        label = tf.image.decode_png(label_name, channels = 1)
        #label = tf.image.resize_images(label, [320, 480])
        label = tf.image.resize_images(label, [522, 775])
        label = tf.to_int64(label > 0)

        return image, label

    def _generate_next_batch(self, image, label, min_queue_examples,
                                         shuffle):
        """Construct a queued batch of index, cont and labels. """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' index + cont + labels from the example queue.
        if shuffle:
            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size = self.batch_size,
                num_threads = self.num_threads,
                capacity = min_queue_examples + 3 * self.batch_size,
                min_after_dequeue = min_queue_examples)
        else:
            images, labels = tf.train.batch(
                [image, label],
                batch_size = self.batch_size,
                num_threads = self.num_threads,
                capacity = min_queue_examples + 3 * self.batch_size)


        return {'images' : images, 'labels' : labels}

    def next_train(self):
        with tf.name_scope('train'), tf.device('/cpu:0'):
            train_data = os.path.join(self.data_dir,'ImageSets', 'Segmentation', 'Cell_small_train.txt')
            if not os.path.exists(train_data):
                print('no train data')
                exit(1)

            images, labels = self.read_file_list(train_data)

            # Create a queue that produces the filenames to read.
            self.filename_queue = tf.train.slice_input_producer([images, labels], num_epochs = self.num_epochs,
                                                            name = 'slice_input_producer')

            # Read examples from files in the filename queue.
            image, label = self.read_and_decode()

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            num_examples_per_epoch = 0.5 * self.batch_size
            min_queue_examples = int(num_examples_per_epoch *
                                     min_fraction_of_examples_in_queue)

            # Generate a batch of images and labels by building up a queue of examples.
            return self._generate_next_batch(image, label, min_queue_examples, shuffle = True)

    def next_test(self):
        with tf.name_scope('test'):
            test_data = os.path.join(self.data_dir,'ImageSets', 'Segmentation', 'Cell_small_test.txt')
            if not os.path.exists(test_data):
                print('no test data')
                exit(1)

            images, labels = self.read_file_list(test_data)
            # Create a queue that produces the filenames to read.
            self.filename_queue = tf.train.slice_input_producer([images, labels], num_epochs = self.num_epochs,
                                                            name = 'slice_input_producer')

            # Read examples from files in the filename queue.
            image, label = self.read_and_decode()


            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            num_examples_per_epoch = 0.5 * self.batch_size
            min_queue_examples = int(num_examples_per_epoch *
                                     min_fraction_of_examples_in_queue)

            # Generate a batch of images and labels by building up a queue of examples.
            return self._generate_next_batch(image, label, min_queue_examples, shuffle = True)

    def next_val(self):
        with tf.name_scope('val'):
            val_data = os.path.join(self.data_dir,'ImageSets', 'Segmentation', 'val.txt')
            if not os.path.exists(val_data):
                print('no val data')
                exit(1)


            images, labels = self.read_file_list(val_data)
            # Create a queue that produces the filenames to read.
            self.filename_queue = tf.train.string_input_producer([images, labels], num_epochs = self.num_epochs,
                                                            name = 'string_input_producer')

            # Read examples from files in the filename queue.
            image, label = self.read_and_decode()


            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            num_examples_per_epoch = 0.5 * self.batch_size
            min_queue_examples = int(num_examples_per_epoch *
                                     min_fraction_of_examples_in_queue)

            # Generate a batch of images and labels by building up a queue of examples.
            return self._generate_next_batch(image, label, min_queue_examples, shuffle = False)



def main(argv = None):
    reader = Reader('/home/bashir/VOC2012', batch_size = 100000)
    images, labels = reader.next_train()

if __name__ == '__main__':
    tf.app.run()
