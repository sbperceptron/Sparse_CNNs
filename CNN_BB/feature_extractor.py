import numpy as np
import glob

class Feature_extractor:
	def __init__(self,obj_path,RFCar,dtype,resolution,fvthresh,x=None,y=None,
		z=None):
		self.obj_path=obj_path
		self.dtype=dtype
		self.resolution=resolution
		self.x=x
		self.y=y
		self.z=z 
		self.fvthresh=fvthresh
		self.RFCar=RFCar

	'''Load the point cloud from the path'''
	def load_pc_from_bin(self,bin_path):
		"""Load PointCloud data from pcd file."""
		obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1,4)
		return obj

	"""Filter camera angles for KiTTI Datasets"""
	def filter_camera_angle(self,pc):
		bool_in = np.logical_and((pc[:, 1] < pc[:, 0] - 0.27), (-pc[:, 1] < pc[:, 0] - 0.27))
		return pc[bool_in]

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
			#   print("data shape: ", data.shape)
			unique,index,inverse,counts=np.unique(data[:,:3],return_index=True,return_inverse=True,return_counts=True,axis=0)
			grid_cells_features=np.zeros((len(unique)-1,6))
			unique=unique[0:len(unique)].astype(int)
			l=len(unique)
			FVS=dict()
			ind0=0
			ind1=1
			intensity_mean=[]
			intensity_var=[]
			counter=0
			for i in range (0,l-1):
				xvalues=data[index[ind0]:index[ind1]][:,3]
				yvalues=data[index[ind0]:index[ind1]][:,4]
				zvalues=data[index[ind0]:index[ind1]][:,5]
				X=np.stack((xvalues,yvalues,zvalues), axis=0)
				# print X.shape
				if X.shape[1]> self.fvthresh:
					counter+=1
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
					# print CL,CP,CS,i_mean_cell,i_variance_cell,1
				ind0+=1
				ind1+=1
			return FVS,counter
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
			coordTuple = (i[0], i[1], i[2])
			FVS_dict[coordTuple]=window[i[0],i[1],i[2],:]
		return FVS_dict

	def feature_extractor(self):
		# loading the point cloud
		if self.dtype=="pcd":
			pc=self.load_pcd_file(self.obj_path)
		elif self.dtype=="bin":
			pc=self.load_pc_from_bin(self.obj_path)
		# The extremes of the input pc
		self.x=(np.amin(pc[:,0]), np.amax(pc[:,0]))
		self.y=(np.amin(pc[:,1]), np.amax(pc[:,1]))
		self.z=(np.amin(pc[:,2]), np.amax(pc[:,2]))
		# Converting the pc file into Feature vector space
		grid=self.raw_to_grid(pc)
		FVS,counter=self.grid_to_FVS(grid)
		# FVS_dict=self.window_array_to_dict(FVS)
		return FVS,counter
