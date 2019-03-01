from train import Train_Model2
import glob
import os
import torch
import torch.nn as nn
import numpy as np

########################################################################################################
########################################################################################################
# paths to 80:20 split data (load the train data)
pcs_orig="./data/kittisplit/train/bin/*.bin"
labels_orig="./data/kittisplit/train/labels/*.txt"
calibs_orig="./data/kittisplit/train/calibs/*.txt"
#######################################################

objects="Pedestrian"#Cyclist Car Pedestrian
#######################################################

#########################################################################################################
#########################################################################################################
# the full point cloud and labels files 
########################################################
full_pcs_bin_paths=pcs_orig
full_pcs_labels_paths=labels_orig
full_pcs_calibs_paths=calibs_orig
full_pcs_bin_paths =glob.glob(full_pcs_bin_paths)
full_pcs_labels_paths = glob.glob(full_pcs_labels_paths)
full_pcs_calibs_paths = glob.glob(full_pcs_calibs_paths)
full_pcs_bin_paths.sort()
full_pcs_labels_paths.sort()
full_pcs_calibs_paths.sort()

full_pcs_bin_paths=full_pcs_bin_paths
full_pcs_labels_paths=full_pcs_labels_paths
full_pcs_calibs_paths=full_pcs_calibs_paths
########################################################

# paths to positive and negative crops of the data
########################################################
car_positives="./data/crops/train/positive/"+objects+"/*.bin"
neg="./data/crops/train/negative/"+objects+"/*.bin"
# hnm_path
hnm_path="./data/crops/train/negative/"+objects+"/hnm/"
#######################################################

# the network Hyper parameters refer to section V C
# The control parameters
###################################################
fvthresh=3 #min number of points for feature extraction
wdthresh=6 # min number of feature vectors per window
#refer to the voting for voting in online pc(Wang and Posner) Section 7 C
resolution=0.2 
# The hnm batch size
batchsizehnm=6 # make it equal to the number of cores -2 on the system
###################################################

# The path where we store the scores and loss values
######################################################
folder=str(fvthresh)+str(wdthresh)
values_path_dir="./data/crops/lossvalues/Pedestrian/2layer/"+folder+"/"
values_file="scoreserror.txt"
if not os.path.exists(values_path_dir):
	os.makedirs(values_path_dir)
if os.path.exists(values_path_dir+values_file):
		ip=raw_input("Do you want to append to the list[Y/n]: ")
		if ip=="N" or ip=="n":
			os.remove(values_path_dir+values_file)
			print("File Removed!")

values_path=values_path_dir+values_file
######################################################

# The Factors affecting the speed of processing
###################################################
batchsize=16 # should be kept constant given by paper
####################################################

# The convergence parameters
####################################################
SGDmomentum=0.9 # Stochastic grdient decsent
L2weightdecay=0.0001
lr=0.001# learning rate
#####################################################

# other parameters
######################################################
# refer to Section V C
pad=1
epochs=100
filter_size=[3,3,3] # Refer to fig 3 in vote3deep
RFCar=[int(2.0/0.2),int(1.0/0.2),int(2.0/0.2)] # # pedestrian
# 95 percentile Receptive field size of the object (Refer to Section 3)
# RFCar=[int(2.0/0.2),int(1.0/0.2),int(2.0/0.2)] # cyclist
N=8 # angular bins in 360/45
x=(0, 80)
y=(-40, 40)
z=(-2.5, 1.5)
######################################################

#####################################################################################################
#####################################################################################################
################################# No pretrained weights #############################################
#####      init the weights and biases based on Delving deep into Rectifiers(He et al.)
#####################################################################################################
########################################################
weights_path="./data/crops/\
weights/"+objects+"/2layer/"+folder+"/"
if not os.path.exists(weights_path):
	os.makedirs(weights_path)
########################################################
num_filters1=8 # Refer to fig 3 in vote3deep
channels1=6 # number of features for each grid cell
wght1=torch.empty(num_filters1,channels1,filter_size[0],filter_size[1],filter_size[2])# fig 3 from paper w1
weights1=nn.init.kaiming_normal_(wght1,mode='fan_in',nonlinearity='relu')
weights1=weights1.numpy()
weights1=weights1.T
b1=np.zeros(num_filters1)

num_filters2=8 # Refer to fig 3 in vote3deep
channels2=8 # number of features for each grid cell
wght2=torch.empty(num_filters2,channels2,filter_size[0],filter_size[1],filter_size[2])# fig 3 from paper w1
weights2=nn.init.kaiming_normal_(wght2,mode='fan_in',nonlinearity='relu')
weights2=weights2.numpy()
weights2=weights2.T
b2=np.zeros(num_filters2)


wght3=torch.empty((RFCar[0])*(RFCar[1])*(RFCar[2])*8,1)# fig 3 from paper w2
weights3=nn.init.kaiming_normal_(wght3,mode='fan_in',nonlinearity='relu')
weights3=weights3.numpy()
b3=np.zeros(1) # page 4 b=0
curr_epoch=0

train_obj=Train_Model2(batchsize,full_pcs_bin_paths, full_pcs_labels_paths,full_pcs_calibs_paths,car_positives,neg,hnm_path,resolution,epochs,\
  lr,SGDmomentum,L2weightdecay,N,weights1,weights2,weights3,b1,b2,b3,RFCar,pad,curr_epoch,channels1,channels2,num_filters1,num_filters2, \
  x,y,z,fvthresh,wdthresh,batchsizehnm, objects, weights_path, values_path)

train_obj.train()
#######################################################################################################
####################### IF YOU HAVE PRETRAINED WEIGHTS ################################################
####################### READ COMMENT ABOVE, AND COMMENT THE NEXT LINES 
####################### Training from a set of known pretrained weights.  #############################
# weights_path="./data/crops/\
# weights/"+objects+"/2layer/"+folder+"/"
# epoch=9
# file_name="epoch_"+str(epoch)+".weights.npz"
# num_filters1=8 # Refer to fig 3 in vote3deep
# channels1=6 # number of features for each grid cell
# num_filters2=8 # Refer to fig 3 in vote3deep
# channels2=8 # number of features for each grid cell
########################################################################################################
# wghts=np.load(weights_path+file_name)

# weights1=wghts['w1']
# weights2=wghts['w2']
# weights3=wghts['w3']
# b1=wghts['b1']
# b2=wghts['b2']
# b3=wghts['b3']
# train_obj=Train_Model2(batchsize,full_pcs_bin_paths, full_pcs_labels_paths,full_pcs_calibs_paths,car_positives,neg,hnm_path,resolution,epochs,\
#   lr,SGDmomentum,L2weightdecay,N,weights1,weights2,weights3,b1,b2,b3,RFCar,pad,epoch,channels1,channels2,num_filters1,num_filters2, \
#   x,y,z,fvthresh,wdthresh,batchsizehnm,objects,weights_path,values_path)
# train_obj.train()
