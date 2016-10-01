import numpy as np
from skimage import io
im = io.imread('birds1.tif')
im_fl = (np.greater(im,128)).astype('float32')

n_image = im_fl.shape[0]
w       = im_fl.shape[1]
assert w == im_fl.shape[2]

data_all = np.reshape(im_fl, (n_image, w, w, 1))
# Shuffle data
np.random.shuffle(data_all)

n_test = n_image // 10
n_train = n_image - n_test

data_train = data_all[0:n_train, :,:,:]
data_test  = data_all[(n_train-1):,:,:,:]

#print data_train.shape
#print data_test.shape

# labels (binary)
cls_train = np.greater(data_train, 0.5).astype('float32')
labels_train = np.concatenate((cls_train,1.0-cls_train),axis=3)
cls_test = np.greater(data_test, 0.5).astype('float32')
labels_test = np.concatenate((cls_test,1.0-cls_test),axis=3)

in_feats     = [1,2,4,8,16,32,64]
out_feats    = [2,4,8,8,16,32,64]

#in_feats     = [1,2,8]
#out_feats    = [2,4,8]

np.savez('silhounette.npz',
	       data_train=data_train,
	       labels_train=labels_train,
	       data_test=data_test,
	       labels_test=labels_test,
           in_feats=in_feats,
           out_feats=out_feats)
