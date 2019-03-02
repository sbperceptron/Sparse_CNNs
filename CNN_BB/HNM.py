import numpy as np
import os,glob
import shutil
import multiprocessing as mp
from models import nn_model2
import time
import ntpath
from readlabels import Read_label_data
from readcalibs import Read_Calib_File,Proj_to_velo

class HNM:
	def __init__(self,w1,w2,w3,b1,b2,b3,full_pcs_bin_paths,full_pcs_labels_paths,
		full_pcs_calibs_paths,neg_root,resolution, RFCar,x,y,z,
		fvthresh,wdthresh,batchsize,objects):
		self.x=x
		self.y=y
		self.z=z
		self.resolution=resolution
		self.w1=w1
		self.w2=w2
		self.w3=w3
		self.b1=b1
		self.b2=b2
		self.b3=b3
		self.full_pcs_bin_paths=full_pcs_bin_paths
		self.full_pcs_labels_paths=full_pcs_labels_paths
		self.full_pcs_calibs_paths=full_pcs_calibs_paths
		self.neg_root=neg_root
		self.RFCar=RFCar
		self.fvthresh=fvthresh
		self.wdthresh=wdthresh
		self.f1=w1.shape[-1]
		self.ch1=w1.shape[-2]
		self.f2=w2.shape[-1]
		self.ch2=w2.shape[-2]
		self.batchsize=batchsize
		self.objects=objects

	'''Crops to file'''
	def crops_to_file(self,crop,root,itr):
		
		name='/'+'%06d' %itr+'_hnm' + '.bin'
		crop.astype(np.float32).tofile(root+name)

	'''Load the point cloud from the path'''
	def load_pc_from_bin(self,bin_path):
		"""Load PointCloud data from pcd file."""
		obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1,4)
		return obj

	"""Filter camera angles for KiTTI Datasets"""
	def filter_camera_angle(self,pc):
		bool_in = np.logical_and((pc[:, 1] < pc[:, 0] - 0.27), (-pc[:, 1] < pc[:, 0] - 0.27))
		return pc[bool_in]

	"""Given the rgbd data, and object bounding box. Crop the rgbd and return the object pointcloud
	free of other points except for the object"""
	def delete_pc_data(self,pc,x_range,y_range,z_range):
		pc_object=pc
		pc_object_x=np.logical_and(pc_object[:,0] > x_range[0], pc_object[:,0]< x_range[1])
		pc_object_y=np.logical_and(pc_object[:,1] > y_range[0], pc_object[:,1]< y_range[1])
		pc_object_z=np.logical_and(pc_object[:,2] > z_range[0], pc_object[:,2]< z_range[1])
		crop=pc_object[np.logical_not(np.logical_and(pc_object_x, np.logical_and(pc_object_y, pc_object_z)))]
		return crop

	"""Create 8 corners of bounding box from bottom center."""
	def get_boxcorners(self,places, rotates, size):
		corners = []
		for place, rotate, sz in zip(places, rotates, size):
			x, y, z = place
			h, w, l = sz

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

	'''remove the objects from the given pointcloud'''
	def remove_objects(self,pc,places,rotates,sizes):
		objs_corners=self.get_boxcorners(places, rotates, sizes)
		for corners in objs_corners:
			corners=corners[np.lexsort((corners[:,2],corners[:,1],corners[:,0]))]
			place1=corners[0]
			place2=corners[-1]
			# print corners
			xbound=(min(place1[0],place2[0]),max(place1[0],place2[0]))
			ybound=(min(place1[1],place2[1]),max(place1[1],place2[1]))
			zbound=(min(place1[2],place2[2]), max(place1[2],place2[2]))
			newpc=self.delete_pc_data(pc,xbound,ybound,zbound)
			pc=newpc

		return newpc

	'''The raw point cloud is converted into grid and then the grid generated is used to create feature vector map for every occupied cell'''
	def raw_to_grid(self,pc):
		resolution=self.resolution
		data=[]
		x=self.x
		y=self.y
		z=self.z
		logic_x = np.logical_and(pc[:, 0] >= x[0], pc[:, 0] < x[1])
		logic_y = np.logical_and(pc[:, 1] >= y[0], pc[:, 1] < y[1])
		logic_z = np.logical_and(pc[:, 2] >= z[0], pc[:, 2] < z[1])
		pointcloud = pc [np.logical_and(logic_x, np.logical_and(logic_y, logic_z))]

		# print "pc:",pointcloud.shape

		for i in pointcloud:
			xBin = ((i[0] - x[0])/resolution).astype(np.int32)
			yBin = ((i[1] - y[0])/resolution).astype(np.int32)
			zBin = ((i[2] - z[0])/resolution).astype(np.int32)
			if (xBin==0):
				xBin=1

			if (yBin==0):
				yBin=1

			if (zBin==0):
				zBin=1

			arr= [xBin,yBin,zBin,i[0],i[1], i[2], i[3]]
			data.append(arr)
		
		data=np.array(data)
		if not data.shape[0]< 2:
			data=data[np.lexsort((data[:,2],data[:,1],data[:,0]))]
		return data

	def grid_to_FVS(self,data):
		x=self.x
		y=self.y
		z=self.z
		RFCar=self.RFCar
		resolution=self.resolution
		FVS_array=np.zeros((int((x[1]-x[0])/0.2),int((y[1]-y[0])/0.2),int((z[1]-z[0])/0.2),6))
		if data.shape[0] > 2:
			unique,index,inverse,counts=np.unique(data[:,:3],axis=0,return_index=True,return_inverse=True,return_counts=True)
			grid_cells_features=np.zeros((len(unique)-1,6))
			unique=unique[0:len(unique)].astype(int)
			l=len(unique)
			FVS=dict()
			ind0=0
			ind1=1
			intensity_mean=[]
			intensity_var=[]
			for i in range (0,l-1):
				xvalues=data[index[ind0]:index[ind1]][:,3]
				yvalues=data[index[ind0]:index[ind1]][:,4]
				zvalues=data[index[ind0]:index[ind1]][:,5]
				X=np.stack((xvalues,yvalues,zvalues), axis=0)
				if X.shape[1]>= self.fvthresh:
					covariance=np.cov(X)
					pearson_corelation_coeff=np.corrcoef(X)
					pearson_corelation_coeff[np.isnan(pearson_corelation_coeff)] = 0
					
					eigen=np.linalg.eigvals(pearson_corelation_coeff)
					eigen.sort()
					eigenvalues=eigen[::-1]
					eigenmean=sum(eigenvalues)/eigenvalues.shape[0]
					
					CL=((eigenvalues[0]-eigenvalues[1])/(3*eigenmean)).astype(np.float16)
					CP=(2*(eigenvalues[1]-eigenvalues[2])/(3*eigenmean)).astype(np.float16)
					CS=(eigenvalues[2]/eigenmean).astype(np.float16)
					
					i_mean_cell=(np.mean(data[index[ind0]:index[ind1]][:,6])).astype(np.float16)
					i_variance_cell=(np.var(data[index[ind0]:index[ind1]][:,6])).astype(np.float16)

					coordTuple = (unique[i][0], unique[i][1], unique[i][2])

					FVS[coordTuple]=[CL,CP,CS,i_mean_cell,i_variance_cell,1]

					
					FVS_array[unique[i][0],unique[i][1],unique[i][2],:]=[CL,CP,CS,i_mean_cell,i_variance_cell,1]
					# print CL,CP,CS,i_mean_cell,i_variance_cell,1
				ind0+=1
				ind1+=1
			return FVS_array
		else:
			return None,None

	'''Function to convert window array to window dictionary'''
	def window_array_to_dict(self,window):
		# we need the xyz values of each feature vector in the FVS space
		FVS_dict=dict()
		loc=np.where(window[:,:,:,5]==1)
		loc=np.hstack((np.array(loc[0]).reshape(-1,1),
			np.array(loc[1]).reshape(-1,1),
			np.array(loc[2]).reshape(-1,1)))
		loc=loc.tolist()
		for i in loc:
			FVS_dict[str(i[0])+" "+str(i[1])+" "+str(i[2])]=window[i[0],i[1],i[2],:]
		return FVS_dict

	'''from the FVS calicluate the score values of the input point cloud'''
	def score_values(self,FVS,label):
		weights1=self.w1
		weights2=self.w2
		weights3=self.w3
		b1=self.b1
		b2=self.b2
		b3=self.b3
		windowsize=self.RFCar
		# instead of looping over the entire of the input
		# look at the locations where we have feature descriptors
		loc=np.where(FVS[:,:,:,5]==1)
		print("\t Number of windows(equal to number of features):",len(loc[0]))
		loc=np.hstack((np.array(loc[0]).reshape(-1,1),
			np.array(loc[1]).reshape(-1,1),
			np.array(loc[2]).reshape(-1,1)))
		loc=loc.tolist()
		count=0
		work=[]
		location=[]
		score_values=[]
		locations=[]
		for i in loc:
			# for each window caliculating the score value
			window=FVS[i[0]-int(windowsize[0]/2):i[0] + int(windowsize[0]/2), 
			i[1]-int(windowsize[1]/2):i[1] + int(windowsize[1]/2),
			i[2] -int(windowsize[2]/2):i[2] + int(windowsize[2]/2),:]
			# also skip the computation if the minimum number of binary occupancy values 
			# in the window is not met
			windowloc= np.where(window[:,:,:,5]==1)
			
			if len(windowloc[0])<self.wdthresh:
				continue
			window=self.window_array_to_dict(window)
			signal=True
			count+=1
			work.append([window,-1,weights1,weights2,weights3,b1,b2,b3,self.RFCar,signal,self.ch1,self.ch2,self.f1,self.f2])
			location.append(i)
			# subjecting each window to the trained model
			# collecting samples for multi processing
			if count%self.batchsize==0:
				pool=mp.Pool(processes=self.batchsize)
				results=pool.map(nn_model2,work) # nn function in the begining of file
				pool.close()
				pool.join()
				# saving the score values
				results=np.array(results)
				for i,result in enumerate(results[:,1]):
					score_values.append(result)
					locations.append(location[i])
				# emptying the work, locations and count values for processing next batch
				work=[]
				location=[]	
				count=0
		return score_values,locations

	"""to find the top 10 score values"""
	def findtop10(self,scores,locations):
		topn_id=sorted(range(len(scores)), key=lambda i: scores[i])[-10:]
		# the locations of those top n
		locations=[locations[i] for i in topn_id]
		scores=[scores[i] for i in topn_id]
		return scores,locations

	def bounding_box(self,l):
		# The boundaries of the window in the  raw point cloud
		xbound=((l[0]-int(self.RFCar[0]/2))*self.resolution+self.x[0],
			(l[0]+int(self.RFCar[0]/2))*self.resolution+self.x[0])
		ybound=((l[1]-int(self.RFCar[1]/2))*self.resolution+self.y[0],
			(l[1]+int(self.RFCar[1]/2))*self.resolution+self.y[0])
		zbound=((l[2]-int(self.RFCar[2]/2))*self.resolution+self.z[0],
			(l[2]+int(self.RFCar[2]/2))*self.resolution+self.z[0])
		return xbound,ybound,zbound

	"""Given the rgbd data, and object bounding box. Crop the rgbd and return the object pointcloud
	free of other points except for the object"""
	def crop_pc_data(self,pc,x_range,y_range,z_range):
		pc_object=pc
		pc_object_x=np.logical_and(pc_object[:,0] > x_range[0], pc_object[:,0]< x_range[1])
		pc_object_y=np.logical_and(pc_object[:,1] > y_range[0], pc_object[:,1]< y_range[1])
		pc_object_z=np.logical_and(pc_object[:,2] > z_range[0], pc_object[:,2]< z_range[1])
		crop=pc_object[np.logical_and(pc_object_x, np.logical_and(pc_object_y, pc_object_z))]
		return crop

	"""Publisher of PointCloud data
	publishes point cloud for every 'k' seconds"""
	def publish_pc(self,pc):
		pub = rospy.Publisher("/points_raw", PointCloud2, queue_size=1000000)
		rospy.init_node("pc2_publisher")
		header = std_msgs.msg.Header()
		header.stamp = rospy.Time.now()
		header.frame_id = "velodyne"
		points = pc2.create_cloud_xyz32(header, pc[:, :3])

		pub.publish(points)
		rospy.sleep(30.)
		# r = rospy.Rate(0.1)
		# while not rospy.is_shutdown():
		# 	pub.publish(points)
		# 	r.sleep()

	'''After every 10 epochs the hard negative mining is performed on the full point clouds(i.e the 80:20 split data) and the top ten
	false postives from each point cloud frame are added to the negative training dataset for the object Refer to section V C'''
	def hnm(self):
		New_NEG_crops=[]
		New_NEG_labels=[]
		count=0
		c1=0
		print("File info [Number of files]:",(len(self.full_pcs_bin_paths),len(self.full_pcs_labels_paths),len(self.full_pcs_calibs_paths)))
		for pc,label,calib in zip(self.full_pcs_bin_paths,self.full_pcs_labels_paths,self.full_pcs_calibs_paths):
			start=time.time()
			c1+=1
			head,tail=ntpath.split(pc)
			print("\n\t processing file", tail)
			pc_bin=self.load_pc_from_bin(pc)
			filtered_bin=self.filter_camera_angle(pc_bin)
			print("\t pointcloud data before deleting objects",filtered_bin.shape)
			calib_obj=Read_Calib_File(calib)
			c= calib_obj.read_calib_file()
			pvelo_obj=Proj_to_velo(c)
			proj_velo = pvelo_obj.proj_to_velo()[:, :3]
			label_obj=Read_label_data(label,objects=self.objects,calib_path=calib, is_velo_cam= True,proj_velo=proj_velo)
			places,rotates,sizes=label_obj.read_label_data()
			
			# locations of objects 
			if type(places) is np.ndarray:
				print("\t Number of objects in the pc:", places.shape[0])
				# before feeding to the network remove the objects from the point cloud 
				pc=self.remove_objects(filtered_bin,places,rotates,sizes)
				print("\t pointcloud data after delete",pc.shape)
			else:
				pc=filtered_bin

			grid=self.raw_to_grid(pc)
			FVS=self.grid_to_FVS(grid)
			
			scores,locations=self.score_values(FVS,label)

			top10sc,top10loc=self.findtop10(scores,locations)

			##################################################################################
			# since we do sparse convolution all the cells in the grid that have a 
			# chance of a car being present will be given high votes
			for i,loc in enumerate(top10loc):
				if top10sc[i]<0:
					continue
				xbound,ybound,zbound=self.bounding_box(loc)
				crop=self.crop_pc_data(pc_bin,xbound,ybound,zbound)
				count+=1
				print("\t The shape of the crop " +str(crop.shape)+" and the score "+str(top10sc[i]))
				self.crops_to_file(crop,self.neg_root,count)
			end=time.time()
			print("\t Time Taken for processing a Pointcloud:", np.round((end-start),2))
		return count

# if __name__ == '__main__':
# 	########################################################################################################
# 	########################################################################################################
# 	# paths to 80:20 split data (load the train data)
# 	pcs_orig="/home/saichand/3D_CNN/kittisplit/train/bin/*.bin"
# 	labels_orig="/home/saichand/3D_CNN/kittisplit/train/labels/*.txt"
# 	calibs_orig="/home/saichand/3D_CNN/kittisplit/train/calibs/*.txt"
# 	neg_root="/home/saichand/3D_CNN/vote3deep2layer/data/crops/train/negative/Cyclist"
# 	#########################################################################################################
# 	#########################################################################################################
	
# 	# the full point cloud and labels files 
# 	full_pcs_bin_paths=pcs_orig
# 	full_pcs_labels_paths=labels_orig
# 	full_pcs_calibs_paths=calibs_orig
# 	full_pcs_bin_paths =glob.glob(full_pcs_bin_paths)
# 	full_pcs_labels_paths = glob.glob(full_pcs_labels_paths)
# 	full_pcs_calibs_paths = glob.glob(full_pcs_calibs_paths)
# 	full_pcs_bin_paths.sort()
# 	full_pcs_labels_paths.sort()
# 	full_pcs_calibs_paths.sort()
	
# 	full_pcs_bin_paths=full_pcs_bin_paths
# 	full_pcs_labels_paths=full_pcs_labels_paths
# 	full_pcs_calibs_paths=full_pcs_calibs_paths

# 	WLOC="/home/saichand/3D_CNN/vote3deep2layer/train/weights/"
# 	file_name="epoch_20.weights.npz"
# 	num_filters1=8 # Refer to fig 3 in vote3deep
# 	channels1=6 # number of features for each grid cell
# 	num_filters2=8 # Refer to fig 3 in vote3deep
# 	channels2=8 # number of features for each grid cell
# 	########################################################################################################
# 	wghts=np.load(WLOC+file_name)

# 	weights1=wghts['w1']
# 	weights2=wghts['w2']
# 	weights3=wghts['w3']
# 	b1=wghts['b1']
# 	b2=wghts['b2']
# 	b3=wghts['b3']

# 	resolution=0.2
# 	RFCar=[int(2.0/0.2),int(1.0/0.2),int(2.0/0.2)] # pedestrian
# 	# RFCar=[int(2.0/0.2),int(1.0/0.2),int(2.0/0.2)] # cyclist
# 	# the x,y and z range of the point clouds
# 	x=(0, 80)
# 	y=(-40, 40)
# 	z=(-2.5, 1.5)
# 	fvthresh=5
# 	wdthresh=2
# 	batchsize=8
# 	new=HNM(weights1,weights2,weights3,b1,b2,b3,full_pcs_bin_paths, 
# 			full_pcs_labels_paths,full_pcs_calibs_paths,neg_root,resolution,
# 			RFCar,x,y,z,fvthresh,wdthresh,batchsize)
# 	FPcount=new.hnm()
