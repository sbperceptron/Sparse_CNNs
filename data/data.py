# step1: Split the data into train and test 80:20
# step 2: Creating the 3D crops of data (negative and positive training 3d crops) and storing the crops.
# equalence on the number of the examples.
# step 3: Creating the datsets of hard, moderate, and easy based on the number of laser readings for each 
# object in the datasets (<50 hard, 50> and <150 moderate, >150 easy) and Storing the three sets into three 
# directories. 
import numpy as np 
import glob, os ,random
from shutil import copyfile, copy
# import rospy, random
# import std_msgs.msg
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2
from help_functions import *
from create_positive import *
from create_negpos_dataset import *
from create_negative import *

'''the function takes in the original paths to the bin, label and calib files to output the train and test datasets 
(80:20 split)''' 
def test_train_split(train_size, test_size, lidar_path, label_path, calib_path):
	lidars=glob.glob(lidar_path)
	labels=glob.glob(label_path)
	calibs=glob.glob(calib_path)
	lidars.sort()
	labels.sort()
	calibs.sort()

	train_lidars=lidars[:int(train_size*len(lidars))]
	test_lidars=lidars[int(train_size*len(lidars)):]
	train_labels=labels[:int(train_size*len(lidars))]
	test_labels=labels[int(train_size*len(lidars)):]
	train_calibs=calibs[:int(train_size*len(lidars))]
	test_calibs=calibs[int(train_size*len(lidars)):]

	return train_lidars,test_lidars, train_labels, test_labels, train_calibs, test_calibs

if __name__== '__main__':
	train_size=0.80
	test_size=1-train_size
	# THE DOWNALOADED KITTI DATASET
	lidar_path="/home/saichand/3D_CNN/KITTI/data_object_velodyne/training/velodyne/*.bin"# LOADING ORIGINAL lidar FILES
	label_path="/home/saichand/3D_CNN/KITTI/label_2/*.txt"# LOADING ORIGINAL LABEL FILES
	calib_path="/home/saichand/3D_CNN/KITTI/calib/training/calib/*.txt"# LOADING ORIGINAL CALIB FILES

	#############################################################################################################################
	# where the final positive and negative  crops will be stored 
	root_main= "/home/saichand/3D_CNN/vote3deep2layer/data/crops"
	# path to train and test sets will be stored. Created by splitting the original data into 80:20 (train and test)
	root="/home/saichand/3D_CNN/kittisplit"
	##############################################################################################################################
	

	# Step1: Splitting the original data into train, test sets (80:20)
	train_lidars_orig, test_lidars_orig, train_labels_orig, test_labels_orig, train_calib_orig, test_calib_orig= test_train_split(train_size, test_size, lidar_path, label_path, calib_path)
	##############################################################################################################################
	# Step2: Creating the positive and negative crops from full point cloud files using the label information
	# firstly creating the folder hierarchy if it doesnt exist
	if not os.path.exists(root+"/train"):
		pths1=list([train_lidars_orig,  train_labels_orig, test_lidars_orig, test_labels_orig , train_calib_orig, test_calib_orig])
		dirs_orig=["/train/bin", "/train/labels","/test/bin", "/test/labels", "/train/calibs","/test/calibs"]
		orig_dirs1=dirs_create(root,dirs_orig)
		print(orig_dirs1)
		to_write=create_dict(orig_dirs1,pths1)
		write_to_file(to_write)
		train_lidars=glob.glob(root+"/train/bin/*.bin")
		train_labels=glob.glob(root+"/train/labels/*.txt")
		train_calibs=glob.glob(root+"/train/calibs/*.txt")
		test_lidars=glob.glob(root+"/test/bin/*.bin")
		test_labels=glob.glob(root+"/test/labels/*.txt")
		test_calibs=glob.glob(root+"/test/calibs/*.txt")
		train_lidars.sort()
		train_labels.sort()
		train_calibs.sort()
		test_lidars.sort()
		test_labels.sort()
		test_calibs.sort()


	# if exists load the bin, label and calib files of the dataset 
	else:
		train_lidars=glob.glob(root+"/train/bin/*.bin")
		train_labels=glob.glob(root+"/train/labels/*.txt")
		train_calibs=glob.glob(root+"/train/calibs/*.txt")
		test_lidars=glob.glob(root+"/test/bin/*.bin")
		test_labels=glob.glob(root+"/test/labels/*.txt")
		test_calibs=glob.glob(root+"/test/calibs/*.txt")
		train_lidars.sort()
		train_labels.sort()
		train_calibs.sort()
		test_lidars.sort()
		test_labels.sort()
		test_calibs.sort()
	
	# checking if the file folders created are empty 
	if  len(train_lidars) == 0 or len(train_labels) == 0 or len(test_lidars) == 0  or len(test_labels) == 0\
	or len(train_calibs) == 0 or len(test_calibs) == 0:
		print "The Files are not copied properly"
	
	# printing number of files stored in the folders
	print "Train (lidars, labels, calibs)->", len(train_lidars), len(train_labels), len(train_calibs)
	print "Test (lidars, labels, calibs)->", len(test_lidars), len(test_labels), len(test_calibs)
	###############################################################################################################################
	################################################################################################################################
	# creates the positive and negative datasets
	# This function calls the create_negpos_dataset.py file which takes the bin,label and calib files for each of train and test
	# and generate the negative and positive crops
	# refer to create_negpos_dataset.py file for details
	create_neg_pos(root_main,train_lidars,train_labels,train_calibs,test_lidars,test_labels,test_calibs)
	###############################################################################################################################
	###############################################################################################################################
	
# rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map velodyne 10
# due to use of fixed size bounding boxes networks can be trained directly on 3D crops of positive 
# and negative examples whose dimensions are equal to the receptive field size specified by the 
# architecture
	

