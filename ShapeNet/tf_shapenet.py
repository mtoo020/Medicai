# ShapeNet

import numpy as np
import tensorflow as tf
import scipy.misc
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
					#initializer=tf.random_normal_initializer(mean=0.0,stddev=1e-12))
							#initializer=tf.constant_initializer(0.0))
		b = tf.get_variable('biases_%d' % (i+1),
			                shape=[n_out],
			                initializer=tf.constant_initializer(1.0))  # small positive bias for ReLU
		c = tf.nn.conv2d(x,w,strides=[1,k,k,1],padding='SAME')
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
	return tf.reduce_mean(losses, reduction_indices=[0,1,2]), sftmx

def shapenet(data, labels, in_feats, out_feats, n):
	a_out = conv_deconv_layer(data, in_feats, out_feats, 0, n)
	ls, sftmx = softmax_ce_across_image(a_out, labels)
	return a_out, ls, sftmx

base_dir = os.path.dirname(os.path.abspath(__file__))

print base_dir

# Load training data
#d = np.load(os.path.join(base_dir,'shapenet.npz'), mmap_mode='r')
#d = np.load(os.path.join(base_dir,'cells.npz'), mmap_mode='r')   # Breast
#d = np.load(os.path.join(base_dir,'SIM.npz'), mmap_mode='r')
d = np.load(os.path.join(base_dir,'silhounette.npz'), mmap_mode='r')
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
assert data_train.shape[0] == labels_train.shape[0]
assert data_test.shape[0] == labels_test.shape[0]
# Make sure that number of input and output features match
assert data_test.shape[3]   == data_train.shape[3]
assert labels_test.shape[3] == labels_train.shape[3]
# Finally, make sure that in_feats and out_feats match number of features
assert in_feats[0] ==  data_train.shape[3]
assert out_feats[0]==labels_train.shape[3]
# -------------------------------------------------------

batch_size = 10

data   = tf.placeholder(shape=[None,n,n, in_feats[0]],dtype=tf.float32)
labels = tf.placeholder(shape=[None,n,n,out_feats[0]],dtype=tf.float32)

#noisy_data = tf.mul(tf.add(data - tf.constant(0.5), tf.random_normal(shape=tf.shape(data),stddev=0.5)), tf.constant(0.3))
#out_imgs, loss, sftmx = shapenet(noisy_data, labels, in_feats, out_feats, n)
out_imgs, loss, sftmx = shapenet(data, labels, in_feats, out_feats, n)
tf.scalar_summary('loss', loss)
#shapenet_tmpl = tf.make_template('Shapenet', shapenet)
#train_loss = shapenet_tmpl(data_train, labels_train, in_feats, out_feats, n)
#test_loss  = shapenet_tmpl(data_test,  labels_test,  in_feats, out_feats, n)

#print a_out
#print train_loss

saver = tf.train.Saver()

#lr = 1e-8  # Breast
lr = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
print 'Setting learning rate: ', lr
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.0,momentum=0.0).minimize(loss)

merged = tf.merge_all_summaries()
test_writer = tf.train.SummaryWriter('/home/bashir/ImageSeg/Medicai/ShapeNet/test')

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	#saver.restore(sess,os.path.join(base_dir,'model/tf_shapenet_trained-404000'))
	#print 'State restored.'
	for epoch in range(0,100000000):
		# Run optimizer
		data_train.shape[0]
		batch = np.random.randint(0,data_train.shape[0],batch_size)
		_ = sess.run(optimizer, feed_dict={data:data_train[batch,:,:,:], labels:labels_train[batch,:,:,:]})
		#print 'Training loss: ', c
		# Print loss on testing set
		if epoch % 200 == 0:
			trn_loss = sess.run(loss, feed_dict={data:data_train, labels:labels_train})
			summary, tst_loss = sess.run([merged, loss], feed_dict={data:data_test, labels:labels_test})
			test_writer.add_summary(summary, epoch)
			print 'Training and testing loss: ', trn_loss, ' ', tst_loss
		# save output images
		if epoch % 1000 == 0:
			im, sm = sess.run([out_imgs, sftmx], feed_dict={data:data_test, labels:labels_test})
			#print im.shape
			#if epoch==0:
			#	scipy.misc.imsave(os.path.join(base_dir,'img/in1_%d.png' % epoch),   np.clip(0.5 + 0.2*ns[0,:,:,0], 0.0, 1.0))
			#	scipy.misc.imsave(os.path.join(base_dir,'img/in2_%d.png' % epoch),   np.clip(0.5 + 0.2*ns[4,:,:,0], 0.0, 1.0))
			scipy.misc.imsave(os.path.join(base_dir,'img/pred1_%d.png' % epoch), sm[0,:,:,0])
			scipy.misc.imsave(os.path.join(base_dir,'img/pred2_%d.png' % epoch), sm[4,:,:,0])
		# Print loss on training set
		if epoch % 1000 == 0:
			saver.save(sess, os.path.join(base_dir,'model/tf_shapenet_trained'),global_step=epoch)

	#arr = np.random.randn(batch_size, n, n, 1)
	#t1 = time.time()
	#sess.run(a_out, feed_dict={data:arr})
	#t2 = time.time()
	#print (t2-t1)/batch_size
	#sess.run(a_out, feed_dict={data:arr})
	#t3 = time.time()
	#print (t3-t2)/batch_size
