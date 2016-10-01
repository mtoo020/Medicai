import numpy as np

data_train   = np.random.randn(10,256,256,1)
data_test    = np.random.randn(5, 256,256,1)
labels_train = np.zeros((10,256,256,5))
labels_test  = np.zeros((5, 256,256,5))
in_feats     = [1,2,4,8,16,32,48,64,80]
out_feats    = [5,6,7,8,16,32,48,64,80]

data_train[:,0:128,0:128,0] = 0.0
data_test[:,0:128,0:128,0] = 0.0
labels_train[:,0:128,0:128,0] = 1.0
labels_test[:,0:128,0:128,0] = 1.0

np.savez('silhounette.npz',
	       data_train=data_train,
	       labels_train=labels_train,
	       data_test=data_test,
	       labels_test=labels_test,
           in_feats=in_feats,
           out_feats=out_feats)
