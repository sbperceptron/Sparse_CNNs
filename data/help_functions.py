import numpy as np 
import glob, os
from shutil import copyfile, copy
# import rospy
# import std_msgs.msg
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2

def get_boxcorners(places, rotates, size):
    """Create 8 corners of bounding box from bottom center."""
    corners = []
    for place, rotate, sz in zip(places, rotates, size):
        # print place
        x, y, z = place
        h, w, l = sz
        #print place,rotate,size
        if l > 10:
            continue

        corner = np.array([
            [x - l / 2., y - w / 2., z],
            [x + l / 2., y - w / 2., z],
            [x - l / 2., y + w / 2., z],
            [x - l / 2., y - w / 2., z + h],
            [x - l / 2., y + w / 2., z + h],
            [x + l / 2., y + w / 2., z],
            [x + l / 2., y - w / 2., z + h],
            [x + l / 2., y + w / 2., z + h],
        ])

        corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate), -np.sin(rotate), 0],
            [np.sin(rotate), np.cos(rotate), 0],
            [0, 0, 1]
        ])

        a = np.dot(corner, rotate_matrix.transpose())
        a += np.array([x, y, z])
        corners.append(a)
    return np.array(corners, dtype=np.float32)
def dirs_create(root,dirs):
    newdirs=[]
    for i in range(0, len(dirs)):
        os.makedirs(root+dirs[i])
        newdirs.append(str(root+dirs[i]))
    return newdirs

def create_dict(newdirs, paths):
    to_write=dict()
    count=0
    for i in newdirs:
        to_write[str(i)]=paths[count]
        count+=1
    return to_write

def write_to_file(to_write):
    # create the main directories
    print "Begining the Copy operation"
    for j in to_write:
        print ".     "
        for k in to_write[str(j)]:
            copy(str(k),j)

    print "Copying Files Done!..."

def proj_to_velo(calib_data):
    """Projection matrix to 3D axis for 3D Label"""
    rect = calib_data["R0_rect"].reshape(3, 3)
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    inv_rect = np.linalg.inv(rect)
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
    return np.dot(inv_velo_to_cam, inv_rect)

def read_label_from_txt(label_path):
    """Read label from txt file."""
    text = np.fromfile(label_path)
    bbox1 = []
    bbox2 = []
    bbox3 = []
    objlabel=[]
    locations=dict()
    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            if not label:
                continue
            label = label.split(" ")
            if (label[0] == "DontCare"):
                continue

            if label[0] == ("Car"): #or ("Truck") or ("Van"):
                #print label[8:15]
                bbox1.append(label[8:15])

            if label[0] == ("Pedestrian"):
                bbox2.append(label[8:15]) 

            if label[0] == ("Cyclist"):
                bbox3.append(label[8:15])
    

    if bbox1:
        locations[str(0)]=bbox1
    if bbox2:
        locations[str(1)]=bbox2
    if bbox3:
        locations[str(2)]=bbox3
    
    if locations:
        return locations
    else:
        return None

def read_calib_file(calib_path):
    """Read a calibration file."""
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if not line or line == "\n":
                continue
            
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def read_labels(label_path, label_type, calib_path=None, is_velo_cam=False, proj_velo=None):
    """Read labels from xml or txt file.
    Original Label value is shifted about 0.27m from object center.
    So need to revise the position of objects.
    """
    label_final=dict()
    if label_type == "txt": #TODO
        locations = read_label_from_txt(label_path)
        if str(0) in locations.keys():
            new_loc1=locations[str(0)]
            data1 = np.array(new_loc1, dtype=np.float32).reshape(-1,7)
            places1,sizes1,rotates1= data1[:,3:6],data1[:, :3],data1[:,6]
            if not places1.any()==None:
                rotates1 = np.pi / 2 - rotates1
                dummy = np.zeros_like(places1)
                dummy = places1.copy()
                if calib_path:
                    places1 = np.dot(dummy, proj_velo.transpose())[:, :3]
                else:
                    places1 = dummy
                if is_velo_cam:
                    places1[:, 0] += 0.27
                label_final[str(0)]=[places1, rotates1, sizes1]

        if str(1) in locations.keys():
            #print(locations[str(1)])
            new_loc2=locations[str(1)]
            data2 = np.array(new_loc2, dtype=np.float32).reshape(-1,7)
            places2,sizes2,rotates2= data2[:,3:6],data2[:, :3],data2[:,6]
            if not places2.any()==None:
                rotates2 = np.pi / 2 - rotates2
                dummy = np.zeros_like(places2)
                dummy = places2.copy()
                if calib_path:
                    places2 = np.dot(dummy, proj_velo.transpose())[:, :3]
                else:
                    places2 = dummy
                if is_velo_cam:
                    places2[:, 0] += 0.27
                label_final[str(1)]=[places2, rotates2, sizes2]

        if str(2) in locations.keys():
            new_loc3=locations[str(2)]
            data3 = np.array(new_loc3, dtype=np.float32).reshape(-1,7)
            places3,sizes3,rotates3= data3[:,3:6],data3[:, :3],data3[:,6]
            if not places3.any()==None:
                rotates3 = np.pi / 2 - rotates3
                dummy = np.zeros_like(places3)
                dummy = places3.copy()
                if calib_path:
                    places3 = np.dot(dummy, proj_velo.transpose())[:, :3]
                else:
                    places3 = dummy
                if is_velo_cam:
                    places3[:, 0] += 0.27
                label_final[str(2)]=[places3, rotates3, sizes3]
    
    if label_final:
        return label_final
    else:
        return []

def load_binfile(bin_path):
    """Load PointCloud data from bin file."""
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def publish_pc(pc):
    """publish two processed data at a time to the ROS node"""
    pub=rospy.Publisher("/First_Frame", PointCloud2, queue_size=1000000)
    rospy.init_node("pc2_publisher")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points = pc2.create_cloud_xyz32(header, pc[:, :3])
    pub.publish(points)
    rospy.sleep(30.)

def publish_two_pc(pc, crop):
    """publish two processed data at a time to the ROS node"""
    pub=rospy.Publisher("/First_Frame", PointCloud2, queue_size=1000000)
    rospy.init_node("pc2_publisher")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points = pc2.create_cloud_xyz32(header, pc[:, :3])

    pub_2=rospy.Publisher("/Last_Frame", PointCloud2, queue_size=1000000)
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points2 = pc2.create_cloud_xyz32(header, crop[:, :3])

    r = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        pub.publish(points)
        pub_2.publish(points2)
        r.sleep()
def filter_camera_angle(places):
    """Filter camera angles for KiTTI Datasets"""
    bool_in = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    # bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
    return places[bool_in]
