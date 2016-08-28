import numpy as np
from skimage import io
im_data   = io.imread('N2DH-SIM-data256.tif')
im_labels = io.imread('N2DH-SIM-labels256.tif')

n_image = im_data.shape[0]
w       = im_data.shape[1]
assert w == im_data.shape[2]


data_all = np.reshape(im_data.astype('float32')/255,     (n_image, w, w, 1))
labl_all = np.reshape(im_labels.astype('float32'),   (n_image, w, w, 1))

shuf = np.arange(0,n_image)
np.random.shuffle(shuf)
data_all = data_all[shuf, :, :, :]
labl_all = labl_all[shuf, :, :, :]

n_test = n_image // 10
n_train = n_image - n_test

data_train = data_all[0:n_train, :,:,:]
data_test  = data_all[(n_train-1):,:,:,:]
labl_train = labl_all[0:n_train, :,:,:]
labl_test  = labl_all[(n_train-1):,:,:,:]

#print data_train.shape
#print data_test.shape

# labels (binary)
cls_train = np.greater(labl_train, 0.5).astype('float32')
labels_train = np.concatenate((cls_train,1.0-cls_train),axis=3)
cls_test = np.greater(labl_test, 0.5).astype('float32')
labels_test = np.concatenate((cls_test,1.0-cls_test),axis=3)

# input size: 512x512
in_feats     = [1,4,8,16,32,64,128,256]
out_feats    = [2,4,8,16,32,64,128,256]


np.savez('SIM.npz',
	       data_train=data_train,
	       labels_train=labels_train,
	       data_test=data_test,
	       labels_test=labels_test,
           in_feats=in_feats,
           out_feats=out_feats)
