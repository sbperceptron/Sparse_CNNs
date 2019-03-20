creating the train and test data.

# objs_receptive_field.py
The main objective of this algorithm is to read in the labels of all the objects in the train full point cloud data and from the places,sizes,rotates caliculate the 95 perecentile size of each object.

# pc_publish.py
Given the point cloud data this algorithm publishes the data as a rostopic which can be visualised using the rviz.

# create_negpos_dataset.py
This is the code used to create the crops data from the full pointcloud KITTI Data.

# Help functions.py
## getboxcorners
Given the places,rotates,sizes of objects finds the 8 corners of the objects and returns them
## proj_to_velo:
Use the calibration data provided with kitti dataset to project the labels into the velodyne frame of reference
## read_label_from_txt:
read the label data from the file

## read_calib_file
read the calibration data from the file
## read_labels
read places,rotates,sizes information from the label data provided by the file read_label_from_txt
## load_binfile
read the pointcloud data from the bin file
## publish_pc
publish the point cloud data from a single publisher node.
## publish_two_pc
publish two point cloud data from two publisher nodes
## filter_camera_angle:
from the full point cloud remove the point data which is not in the camera field of view

# createnegposfn.py
the set of functions common to creation of data for the model to train.

## translate:
for translating the object by a fixed distance
## zaxisrotate
rotating the object about zaxis by a fixed angle
## augmentdata:
use combination of translation and rotation to create augmented data
## crops_to_file:
Store all the crops created by the algortihms to the file and also augment the positive crops 
## crops_to_file_test:
Same as above but for the test examples
## crop_object_from_pc_data:
Cropping object from full point cloud and returning the crop or returning the pointcloud without crop
## find_object_ranges:
Given the places,rotates,sizes create the object ranges from
## Object_ranges: 
For all the objects in the point cloud  find the places,rotates,sizes of objects in there and then use the find_object_ranges function to find all the ranges

# data.py
The main function used to create the test,train split of the full pointcloud data and the positive and negative crops from the full point clouds

Dependent on the create_negpos_dataset.py file

