import rospy
import numpy as np
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import glob

# terminal 1 python pc_publish.py
# terminal 2 roscore
# terminal 3 rosrun rviz rviz
# terminal 4 rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map velodyne 10
def publish_pc(object1):
	pub=rospy.Publisher("/pointcloud1", PointCloud2, queue_size=1000000)
	rospy.init_node("pc2_publisher")
	header = std_msgs.msg.Header()
	header.stamp = rospy.Time.now()
	header.frame_id = "velodyne"
	points = pc2.create_cloud_xyz32(header, object1[:, :3])
	r = rospy.Rate(0.1)
	while not rospy.is_shutdown():
		pub.publish(points)
		r.sleep()

def publish_three_pc(object1, object2):#, pc3):
	"""publish two processed data at a time to the ROS node"""
	pub=rospy.Publisher("/pointcloud1", PointCloud2, queue_size=1000000)
	rospy.init_node("pc2_publisher")
	header = std_msgs.msg.Header()
	header.stamp = rospy.Time.now()
	header.frame_id = "velodyne"
	points = pc2.create_cloud_xyz32(header, object1[:, :3])

	pub_2=rospy.Publisher("/pointcloud2", PointCloud2, queue_size=1000000)
	header = std_msgs.msg.Header()
	header.stamp = rospy.Time.now()
	header.frame_id = "velodyne"
	points2 = pc2.create_cloud_xyz32(header, object2[:, :3])

	# pub_3=rospy.Publisher("/pc3", PointCloud2, queue_size=1000000)
	# header = std_msgs.msg.Header()
	# header.stamp = rospy.Time.now()
	# header.frame_id = "velodyne"
	# points3 = pc2.create_cloud_xyz32(header, pc3[:, :3])

	r = rospy.Rate(0.1)
	while not rospy.is_shutdown():
		pub.publish(points)
		pub_2.publish(points2)
		#pub_3.publish(points3)
		r.sleep()

def load_binfile(bin_path):
	"""Load PointCloud data from bin file."""
	obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
	return obj

if __name__== '__main__':
	# paths to files which you want to view
	pointcloud1=glob.glob("/home/saichand/3D_CNN/vote3deep2layer/data/crops/test/split/positive/Cyclist/medium/*.bin")
	pointcloud2=glob.glob("/home/saichand/3D_CNN/vote3deep2layer/data/crops/test/split/negative/Cyclist/medium/*.bin")
	
	#pc3="/home/saichand/3D_CNN/python/basemodel/data/KITTI/crops/train/negative/Pedestrian/000000000.bin"
	for i,j in zip(pointcloud1,pointcloud2):
		pospc=load_binfile(i)
		negpc=load_binfile(j)
	
	print "Positive", pospc.shape
	print "Negative", negpc.shape
	
	publish_three_pc(pospc[:,:3], negpc[:,:3])#, pc3)