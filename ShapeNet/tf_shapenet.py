# ShapeNette

import numpy as np
import tensorflow as tf 
import scipy.misc
import time
import os 

# TODO: fix dropout

def conv_layer(x, hidden_featl, k_sizes, dropout_frac):
	for i in range(len(hidden_featl)-1):
		n_in  = hidden_featl[i]
		n_out = hidden_featl[i+1]
		k = k_sizes[i]
		w = tf.get_variable('weights_%d' % (i+1),
			                shape=[k,k,n_in,n_out],
		                	initializer=tf.contrib.layers.xavier_initializer())
		                    #initializer=tf.random_normal_initializer(mean=1e-8,stddev=1e-8))
							#initializer=tf.constant_initializer(0.0))
		b = tf.get_variable('biases_%d' % (i+1),
			                shape=[n_out],
			                initializer=tf.constant_initializer(0.0))  # small positive bias for ReLU
		c = tf.nn.conv2d(x,w,strides=[1,k,k,1],padding='VALID')
		x = tf.nn.relu(c + b)
	#return tf.nn.dropout(x, dropout_frac)
	return x

def deconv_layer(x, y, hidden_featl, dropout_frac, new_h, new_w):
	n = len(hidden_featl)
	z = tf.concat(3, [x, y])
	z = conv_layer(z, hidden_featl, [1 for i in range(n-1)], dropout_frac)
	return tf.image.resize_images(z, new_h, new_w, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def conv_deconv_layer(a, in_feats, out_feats, i, sz):
	if len(in_feats)==1:
		return a  # TODO: do some MLPs on a
	else:
		with tf.variable_scope('conv_%d' % (i+1)):
			a = conv_layer(a, [in_feats[0], in_feats[1], in_feats[1]], [2, 1], 0.1)
			#print a
		b = conv_deconv_layer(a, in_feats[1:], out_feats[1:], i+1, sz//2)
		with tf.variable_scope('deconv_%d' % (i+1)):
			#print b
			return deconv_layer(a, b, [in_feats[1]+out_feats[1], out_feats[0]], 0.1, sz, sz)

def softmax_ce_across_image(logits, labels):
	p = tf.exp(logits)
	sftmx = tf.div(p, tf.reduce_sum(p, reduction_indices=[3], keep_dims=True))
	losses = -tf.reduce_sum(labels * tf.log(sftmx), reduction_indices=[3], keep_dims=False)
	return tf.reduce_mean(losses, reduction_indices=[0,1,2])

def shapenet(data, labels, in_feats, out_feats, n):
	a_out = conv_deconv_layer(data, in_feats, out_feats, 0, n)
	return a_out, softmax_ce_across_image(a_out, labels)

#base_dir = '/data/ShapeNet'
base_dir = '/media/PQI/ShapeNet'

# Load training data
d = np.load(os.path.join(base_dir,'shapenet.npz'), mmap_mode='r')
data_train   = d['data_train']
data_test    = d['data_test']
labels_train = d['labels_train']
labels_test  = d['labels_test']
in_feats     = d['in_feats']  
out_feats    = d['out_feats'] 

# -------------------------------------------------------
# Data sanity checks
# Make sure image size is the same across all data
n = data_train.shape[1]
assert data_train.shape[1]   == data_train.shape[2]
assert data_train.shape[1]   == n
assert data_test.shape[1]    == data_test.shape[2]
assert data_test.shape[1]    == n
assert labels_train.shape[1] == labels_train.shape[2]
assert labels_train.shape[1] == n
assert labels_test.shape[1]  == labels_test.shape[2]
assert labels_test.shape[1]  == n
# Make sure training data and labels have same batch size
train_batch_size = data_train.shape[0]
assert train_batch_size == labels_train.shape[0]
test_batch_size  = data_test.shape[0]
assert test_batch_size == labels_test.shape[0]
# Make sure that number of input and output features match
assert data_test.shape[3]   == data_train.shape[3]
assert labels_test.shape[3] == labels_train.shape[3]
# Finally, make sure that in_feats and out_feats match number of features
assert in_feats[0] ==  data_train.shape[3]
assert out_feats[0]==labels_train.shape[3]
# -------------------------------------------------------


data   = tf.placeholder(shape=[None,n,n, in_feats[0]],dtype=tf.float32)
labels = tf.placeholder(shape=[None,n,n,out_feats[0]],dtype=tf.float32)

noisy_data = tf.add(tf.mul(data - tf.constant(0.5),tf.constant(0.8)), tf.random_normal(shape=tf.shape(data),stddev=0.2))
out_imgs, loss = shapenet(noisy_data, labels, in_feats, out_feats, n)
#test1 = conv_layer(x, hidden_featl, k_sizes, dropout_frac)

#shapenet_tmpl = tf.make_template('Shapenet', shapenet)
#train_loss = shapenet_tmpl(data_train, labels_train, in_feats, out_feats, n)
#test_loss  = shapenet_tmpl(data_test,  labels_test,  in_feats, out_feats, n)

#print a_out
#print train_loss

saver = tf.train.Saver()

lr = 1e-6
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
print 'Setting learning rate: ', lr
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.0,momentum=0.0).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	#saver.restore(sess,os.path.join(base_dir,'model/tf_shapenet_trained-7500'))
	#print 'State restored.'
	for epoch in range(7500,1000000):
		# Run optimizer
		_ = sess.run(optimizer, feed_dict={data:data_train, labels:labels_train})
		#print 'Training loss: ', c
		# Print loss on testing set
		if epoch % 20 == 0:
			trn_loss = sess.run(loss, feed_dict={data:data_train, labels:labels_train})
			tst_loss = sess.run(loss, feed_dict={data:data_test, labels:labels_test})
			print 'Training and testing loss: ', trn_loss, ' ', tst_loss
		# save output images
		if epoch % 50 == 0:
			ns, im = sess.run([noisy_data, out_imgs], feed_dict={data:data_test, labels:labels_test})
			#print im.shape
			if epoch==0:
				scipy.misc.imsave(os.path.join(base_dir,'img/in1_%d.png' % epoch),   np.clip(0.5 + 0.2*ns[0,:,:,0], 0.0, 1.0))
				scipy.misc.imsave(os.path.join(base_dir,'img/in2_%d.png' % epoch),   np.clip(0.5 + 0.2*ns[4,:,:,0], 0.0, 1.0))
			scipy.misc.imsave(os.path.join(base_dir,'img/pred1_%d.png' % epoch), (im[0,:,:,0] < im[0,:,:,1]).astype('float32'))
			scipy.misc.imsave(os.path.join(base_dir,'img/pred2_%d.png' % epoch), (im[4,:,:,0] < im[4,:,:,1]).astype('float32'))
		# Print loss on training set
		if epoch % 100 == 0:
			saver.save(sess, os.path.join(base_dir,'model/tf_shapenet_trained'),global_step=epoch)

	#arr = np.random.randn(batch_size, n, n, 1)
	#t1 = time.time()
	#sess.run(a_out, feed_dict={data:arr})
	#t2 = time.time()
	#print (t2-t1)/batch_size
	#sess.run(a_out, feed_dict={data:arr})
	#t3 = time.time()
	#print (t3-t2)/batch_size