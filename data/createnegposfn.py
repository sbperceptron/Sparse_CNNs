from help_functions import *
# from pc_publish import *
import random
'''section 5 subsection c'''
'''
the training data is augmented by translating the original 
front facing positive training examples by a distance smaller than the 
size of the 3D grid cells
Translating the object by 0.2m'''
def translate(cr,whichaxis,value):
	cr=np.array(cr)
	new=np.zeros((cr.shape))
	if whichaxis == '0':
		new[:,0]=cr[:,0]+value
		new[:,1:4]=cr[:,1:4]
	if whichaxis == '1':
		new[:,1]=cr[:,1]+value
		new[:,0:1]=cr[:,0:1]
		new[:,2:4]=cr[:,2:4]
	if whichaxis == '2':
		new[:,2]=cr[:,2]+value
		new[:,0:2]=cr[:,0:2]
		new[:,3:4]=cr[:,3:4]
	return new

'''and randomly rotaing them that is smaller than the angle of angular bins.
the angle of the bins 360/8 from voting for voting paper section 7 subsection c
rotating the objects about the z axis
this is rotaing the object about the main axis of the point cloud
for augmenting the data this should do the work'''
def zaxis_rotate(pc,angle):
	pc=np.array(pc)
	"""Rotate all the points about z by a fixed angle"""
	Rz=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0], [0,0,1]]
	zrotation=[]
	final_rot=np.zeros((pc.shape))
	rot_pc=pc[:,:3].dot(Rz)
	final_rot[:,:3]=rot_pc
	final_rot[:,3]=pc[:,3]
	return final_rot

'''Augmenting the training data offline by translating the original pc by 0.2m and then randomly rotating it by an angle
smaller than the resolution of the angular bins'''
def augmentdata(crop):

	aug_crops=[]
	aug_tra_crops=[]
	aug_tra_crops.append(crop)
	trans=random.randint(0,4)
	trans_vlue=[0.2,-0.2]
	for tr in range(0,trans):
		value=random.randint(0,1)
		axis=random.randint(0,2)
		new=translate(crop,str(axis),trans_vlue[value])
		aug_tra_crops.append(new)
	aug_rot_crops=[]
	for crop1 in aug_tra_crops:
		angle=round(random.uniform(0,45),2)
		rot=zaxis_rotate(crop1,angle)
		aug_rot_crops.append(rot)
	augdata=aug_rot_crops+aug_tra_crops
	return np.array(augdata)

'''writing the cropped objects to file'''
def crops_to_file_train(crops, dirs, keys1, keys2,signal):
	count=0
	for key1 in keys1:
		for key2 in keys2:
			itr=0
			dire=dirs[count]
			crop_pcs=crops[str(key1)][str(key2)]
			for i in range (0, len(crop_pcs)):
				if signal=='positive':
					# print crop_pcs[i].astype(np.float16).shape
					augmentedcrops=augmentdata(crop_pcs[i])
					for j in augmentedcrops:
						name='/'+'%06d' %itr + '.bin'
						j.astype(np.float32).tofile(dire+name)
						itr+=1
				elif signal=='negative':
					# print crop_pcs[i].shape
					name='/'+'%06d' %itr + '.bin'
					crop_pcs[i].astype(np.float32).tofile(dire+name)
					itr+=1
			count+=1

'''writing the cropped objects to file'''
def crops_to_file_test(crops, dirs, keys1, keys2, keys3, signal):
	count=0
	for key1 in keys1:
		for key2 in keys2:
			for key3 in keys3:
				itr=0
				dire=dirs[count]
				crop_pcs=crops[str(key1)][str(key2)][str(key3)]
				for i in range (0, len(crop_pcs)):
					if signal=='positive':
						# print crop_pcs[i].astype(np.float16).shape
						augmentedcrops=augmentdata(crop_pcs[i])
						for j in augmentedcrops:
							name='/'+'%06d' %itr + '.bin'
							j.astype(np.float32).tofile(dire+name)
							itr+=1
					elif signal=='negative':
						# print crop_pcs[i].shape
						name='/'+'%06d' %itr + '.bin'
						crop_pcs[i].astype(np.float32).tofile(dire+name)
						itr+=1
				count+=1

'''given the point cloud and the min and max extrwemes of the point cloud 
return the cropped point cloud'''
def crop_object_from_pc_data(pc,r, signal2):
	"""Given the Lidar data, and object bounding box. Crop the Lidar and return the object pointcloud
	free of other points except for the object"""
	pc_object=pc
	x_range=(r[0] , r[1])
	y_range=(r[2] , r[3])
	z_range=(r[4] , r[5])

	pc_object_x=np.logical_and(pc_object[:,0] > x_range[0], pc_object[:,0]< x_range[1])
	pc_object_y=np.logical_and(pc_object[:,1] > y_range[0], pc_object[:,1]< y_range[1])
	pc_object_z=np.logical_and(pc_object[:,2] > z_range[0], pc_object[:,2]< z_range[1])

	if signal2 == 'remcrop':
		crop=pc_object[np.logical_and(np.logical_not(pc_object_x), np.logical_and(np.logical_not(pc_object_y), np.logical_not(pc_object_z)))]
		
	elif signal2 == 'crop':
		crop=pc_object[np.logical_and(pc_object_x, np.logical_and(pc_object_y, pc_object_z))]
	return crop

'''given a set of places ,sizes and rotates return the min and max ranges of the object '''
def findobjranges(mainobj_places,mainobj_rotates,mainobj_sizes):
	objranges=dict()
	labels_objs=[]
	# from the labels of the files finding the range in which the object is located
	# first find the corners of the object in point cloud
	objs_corners=get_boxcorners(mainobj_places, mainobj_rotates, mainobj_sizes)
	count=0
	for corners in objs_corners:
		# print corners.shape
		corners=corners[np.lexsort((corners[:,2],corners[:,1],corners[:,0]))]
		place1=corners[0]
		place2=corners[-1]
		range_object=np.array([place1, place2]).ravel()
		_range=[range_object[0],range_object[3], range_object[1], range_object[4], range_object[2], range_object[5]]
		objranges[str(count)]=_range
		count+=1
	return objranges

# reading all the object labels and finding the extremes of each object in the point cloud
# the places, sizes are used to caliculate the minimum and maximum exteremes of each object
def object_ranges(dict_objects):
	copy=dict_objects

	# firstly checking if there are objects of a particular class in the given set of labels
	if str(0) in dict_objects.keys():
		[carsplaces,carsrotates,carssizes]=dict_objects[str(0)]
	else:
		[carsplaces,carsrotates,carssizes]=[None, None, None]
	
	if str(1) in dict_objects.keys():
		[pedsplaces,pedsrotates,pedssizes]=dict_objects[str(1)]
	else:
		[pedsplaces,pedsrotates,pedssizes]=[None, None, None]
	
	if str(2) in dict_objects.keys():
		[cycsplaces,cycsrotates,cycssizes]=dict_objects[str(2)]
	else:
		[cycsplaces,cycsrotates,cycssizes]=[None, None, None]
	
	# Now if there exists a label an object we find the ranges of the object in pc
	
	if not isinstance(carsplaces, type(None)) and carsplaces.size:
		carranges=findobjranges(carsplaces,carsrotates,carssizes)
	else: 
		carranges=None

	if not isinstance(pedsplaces, type(None)) and pedsplaces.size:
		pedranges=findobjranges(pedsplaces,pedsrotates,pedssizes)
	else:
		pedranges=None
	
	if not isinstance(cycsplaces, type(None)) and cycsplaces.size:
		cycranges=findobjranges(cycsplaces,cycsrotates,cycssizes)
	else:
		cycranges=None
	
	objectranges={'carranges':carranges, 'pedranges':pedranges, 'cycranges':cycranges}
	
	# then return the ranges for each object in the point cloud
	return objectranges