from help_functions import *
# from pc_publish import *
from createnegposfn import *
num_goodPOS_crops=0
num_goodGT_crops=0
from itertools import chain

'''The positive crops are cropped according to 95 percentile object size of the dataset'''
def new_range(range_object,objsz,signal):
	
	# the object size is of the format of height width and length 
	if signal=='POS':
		xdiff=objsz[2]-(range_object[1]-range_object[0])
		ydiff=objsz[1]-(range_object[3]-range_object[2])
		zdiff=objsz[0]-(range_object[5]-range_object[4])

		if xdiff > 0:
			range_object[0]=range_object[0]-xdiff/2 if range_object[0]<0 else range_object[0]-xdiff/2	
			range_object[1]=range_object[1]+xdiff/2 if range_object[1]<0 else range_object[1]+xdiff/2
		
		if ydiff >0:
			range_object[2]=range_object[2]-ydiff/2 if range_object[2]<0 else range_object[2]-ydiff/2
			range_object[3]=range_object[3]+ydiff/2 if range_object[3]<0 else range_object[3]+ydiff/2

		if zdiff >0:
			range_object[4]=range_object[4]-zdiff/2 if range_object[4]<0 else range_object[4]-zdiff/2
			range_object[5]=range_object[5]+zdiff/2 if range_object[5]<0 else range_object[5]+zdiff/2
			
		return range_object

'''for each label the object is cropped from the original pc'''
def create_positives(to_crop, maps, obj_size, signal):
	crops=[]
	for key in maps.keys():
		pc=to_crop
		r=maps[key]
		newr=new_range(r, obj_size, signal)
		crop=crop_object_from_pc_data(pc, newr, signal2='crop')
		crops.append(crop)
	return crops

'''delete the positive crops with zero pints in the point cloud'''
def delzeropc(list1):
	# firstly getting the shape of pint clouds i.e the number of points
	list1shapes=[k.shape[0] for k in list1]
	# number of point clouds with less than zero points
	newlist=[]
	for i in range(0,len(list1)):
		if list1shapes[i]>=10:
			newlist.append(list1[i])
	zeroloc_id=len(list1)-len(newlist)
	return newlist,zeroloc_id

'''to this function the folders to where the crop files need to be stored are passed along 
with original point cloud files path 
pths-> the point cloud files
root-> the main directory 
dirs_crops-> the crops diercetories where the cropped positives will be stored
dirs_labels-> the path to whre the final labels will be stored
objsizes-> the size of each positive crop '''

def create_positive_dataset(pths, root, dirs_crops,objssizes):

	iterations=len(pths)/3
	count=0
	zeroloc=0

	carsize=objssizes['Car_size:']
	pedsize=objssizes['Ped_size:']
	cycsize=objssizes['Cyc_size:']

	crops={'train':{'cars':[], 'peds':[],'cycs':[]},\
			'test':{'cars':[], 'peds':[],'cycs':[]}}
	cropskeys=['train','test']
	
	for itr in range(0, iterations):
		pos=crops[cropskeys[itr]]
		for i,j,k in zip(pths[itr*3], pths[itr*3+1], pths[itr*3+2]):
			count+=1
			# reading the point cloud file from the path
			pc=load_binfile(i)
			# filtering the point clouds which fall into the camera angle
			pc=filter_camera_angle(pc)
			# reading the calib file
			calib = read_calib_file(k)
			# projecting the velo
			proj_velo = proj_to_velo(calib)[:, :3]
			label_type="txt"
			# For each and every object in the point cloud the label information 
			final_label= read_labels(j, label_type, calib_path=calib, is_velo_cam=True, proj_velo=proj_velo)
			objectsranges=object_ranges(final_label)
			# create positives
			if not objectsranges == None:
				if objectsranges['carranges'] != None:
					cars=objectsranges['carranges']
					car_POS = create_positives(pc,cars,carsize, signal='POS')
					pos['cars'].append(car_POS)

				if objectsranges['pedranges'] != None:
					peds=objectsranges['pedranges']
					ped_POS= create_positives(pc,peds,pedsize, signal='POS')
					pos['peds'].append(ped_POS)
					
				if objectsranges['cycranges'] != None:
					cycs=objectsranges['cycranges']
					cyc_POS= create_positives(pc,cycs, cycsize,signal ='POS')
					pos['cycs'].append(cyc_POS)
										
		pos['cars']=[k for sublist in pos['cars'] for k in sublist]
		#delete the zero entries
		pos['cars'], crdel=delzeropc(pos['cars'])

		pos['peds']=[k for sublist in pos['peds'] for k in sublist]
		#delete the zero entries
		pos['peds'],pddel=delzeropc(pos['peds'])

		pos['cycs']=[k for sublist in pos['cycs'] for k in sublist]
		#delete the zero entries
		pos['cycs'],cydel=delzeropc(pos['cycs'])
				
		crops[cropskeys[itr]]=pos
		
		zeroloc+=(crdel+pddel+cydel)

	orig_dirs2=dirs_create(root,dirs_crops)
	keys1=['train', 'test']
	keys2=['cars','peds','cycs']

	crops_to_file_train(crops,orig_dirs2,keys1,keys2,signal='positive')

	
	
	