import os
import glob
from test import Test_Model1,Test_Model2
import numpy as np

# WLOC="/home/saichand/3D_CNN/vote3deep2layer/train/weights/"
WLOC="../crop/weights/cyclists/2layer/feb19/"

dst_path="../crop/results/cyclists/2layer/feb19/"
	# WLOC="/home/saichand/3D_CNN/vote3deep2layer/train_singlelayer/weights/"
	# dst_path="results/"
if not os.path.exists(dst_path):
	os.makedirs(dst_path)

curr_epoch=54
file_name="epoch_"+str(curr_epoch)+".weights.npz"
# Cyclist
objects_pos_easy_pcd="../crop/data/crops/test/split/positive/Cyclist/easy/*.bin"
objects_pos_medium_pcd="../crop/data/crops/test/split/positive/Cyclist/medium/*.bin"
objects_pos_hard_pcd="../crop/data/crops/test/split/positive/Cyclist/hard/*.bin"
objects_neg_easy_pcd="../crop/data/crops/test/split/negative/Cyclist/easy/*.bin"
objects_neg_medium_pcd="../crop/data/crops/test/split/negative/Cyclist/medium/*.bin"
objects_neg_hard_pcd="../crop/data/crops/test/split/negative/Cyclist/hard/*.bin"
RFCar=[int(2.0/0.2),int(1.0/0.2),int(2.0/0.2)] # cyclist
# Pedestrian
# objects_pos_easy_pcd="../crop/data/crops/test/split/positive/Pedestrian/easy/*.bin"
# objects_pos_medium_pcd="../crop/data/crops/test/split/positive/Pedestrian/medium/*.bin"
# objects_pos_hard_pcd="../crop/data/crops/test/split/positive/Pedestrian/hard/*.bin"
# objects_neg_easy_pcd="../crop/data/crops/test/split/negative/Pedestrian/easy/*.bin"
# objects_neg_medium_pcd="../crop/data/crops/test/split/negative/Pedestrian/medium/*.bin"
# objects_neg_hard_pcd="../crop/data/crops/test/split/negative/Pedestrian/hard/*.bin"
# RFCar=[int(2.0/0.2),int(1.0/0.2),int(2.0/0.2)] # pedestrian
########################################################################################################
# the weights for a single layer model
# wghts=np.load(WLOC+file_name)
# weights1=wghts['w1']
# weights2=wghts['w2']
# b1=wghts['b1']
# b2=wghts['b2']
# num_filters1=8 # Refer to fig 3 in vote3deep
# channels1=6 # number of features for each grid cell
# weights for a two layer model
wghts=np.load(WLOC+file_name)
weights1=wghts['w1']
weights2=wghts['w2']
weights3=wghts['w3']
b1=wghts['b1']
b2=wghts['b2']
b3=wghts['b3']
num_filters1=8 # Refer to fig 3 in vote3deep
channels1=6 # number of features for each grid cell
num_filters2=8 # Refer to fig 3 in vote3deep
channels2=8 # number of features for each grid cell

pad=1
batchsize=16
epochs=100
SGDmomentum=0.9 # Stochastic grdient decsent
L2weightdecay=0.0001
lr=0.001# learning rate
filter_size=[3,3,3] # Refer to fig 3 in vote3deep

resolution=0.2 
N=8 # angular bins in 360/45
fvthresh=8

ob1=Test_Model2(batchsize,objects_pos_easy_pcd,objects_pos_medium_pcd,objects_pos_hard_pcd,objects_neg_easy_pcd,objects_neg_medium_pcd,
	objects_neg_hard_pcd,weights1,weights2,weights3,b1,b2,b3,RFCar,channels1,channels2,
	num_filters1,num_filters2,curr_epoch,dst_path,fvthresh)
ob1.test()