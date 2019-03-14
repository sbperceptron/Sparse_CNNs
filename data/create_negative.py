from help_functions import *
# from pc_publish import *
from createnegposfn import *
import random

def check(r, ranges):
	 
	for _range in ranges:
		#print _range
		[x1,x2,y1,y2,z1,z2]=_range
		if (r[0] > x1 and r[0] <x2) or (r[1] > x1 and r[1] <x2):
			boolop=True
			#print(boolop)
		elif (r[2] > y1 and r[2] <y2) or (r[3] > y1 and r[3] <y2):
			boolop=True
			#print(boolop)
		elif (r[4] > z1 and r[4] <z2) or (r[5] > z1 and r[5] <z2):
			boolop=True
			#print(boolop)
		else:
			boolop= False
			#print boolop
	return boolop


def create_negatives_split(to_crop, ranges, cropsize, size1=None, size2=None,emh=None):
	count=1
	numpoints_counted=0
	
	while count  > 0:
		point=to_crop[random.randint(0,len(to_crop)-1)]
		numpoints_counted+=1
		# TODO: take the point as the center of the crop rather than one extemity
		range_object1=np.array([point[:3], point[:3]+cropsize]).ravel()
		r=[range_object1[0],range_object1[3], range_object1[1], range_object1[4], range_object1[2], range_object1[5]]
		bl=check(r,ranges)
		if bl == False:
			crop=crop_object_from_pc_data(to_crop,r, signal2='crop')
			# emh denotes the easy ,medium and hard classes
			# the size1,size2 and size3 indicate the theshold number of points 
			# for the respective classes
			if emh==1:
				if crop.shape[0] > size1:
					crops=crop
					count-=1
			if emh==2:
				if crop.shape[0] > size1 and crop.shape[0] < size2:
					crops=crop
					count-=1
			if emh==3:
				if crop.shape[0] > size1 and crop.shape[0] < size2:
					crops=crop
					count-=1
			# controlling the nmber of times the while loop is processed.
			# after searching for a number specified below if we are not able to find
			# the pc we exit the loop
		if numpoints_counted==500:
			crops=None
			numpoints_counted=0
			break
	return crops

def create_negatives(to_crop, ranges, cropsize):
	count=1
	numpoints_counted=0
	
	while count  > 0:
		point=to_crop[random.randint(0,len(to_crop)-1)]
		numpoints_counted+=1
		range_object1=np.array([point[:3], point[:3]+cropsize]).ravel()
		r=[range_object1[0],range_object1[3], range_object1[1], range_object1[4], range_object1[2], range_object1[5]]
		bl=check(r,ranges)
		if bl == False:
			crop=crop_object_from_pc_data(to_crop,r, signal2='crop')
			if crop.shape[0] > 10:
				crops=crop
				count-=1
			# controlling the nmber of times the while loop is processed.
			# after searching for a number specified below if we are not able to find
			# the pc we exit the loop
		if numpoints_counted==500:
			crops=None
			numpoints_counted=0
			break
	return crops
 
'''function to create negative crops equal to the number of positive crops'''
# TODO: why do we have the size1,size2,signal,emh argumnets for this functions
# get rid if they are redundant
def object_crops(pths,number_objects,objsize,code,size1=None, size2=None,signal=False,emh=None):
	# since the object size is in the format height width and length we will change that to
	# length, width and height format first before passing
	objsize=[objsize[2],objsize[1],objsize[0]]
	numberofobjects=0
	cropsofobject=[]
	while numberofobjects < number_objects:
		ra=random.randint(0, len(pths[0])-1)
		i=pths[0][ra]
		j=pths[1][ra]
		k=pths[2][ra]
		pc=load_binfile(i)
		pc=filter_camera_angle(pc)
		calib = read_calib_file(k)
		proj_velo = proj_to_velo(calib)[:, :3]
		label_type="txt"
		final_label= read_labels(j, label_type, calib_path=calib, is_velo_cam=True, proj_velo=proj_velo)
		dict_objects=final_label
		# all the objects in the pc and their ranges
		objs=[]
		for key in dict_objects.keys():
			objs.append(dict_objects[key])
		
		for i in range(0, len(objs)):
			if i ==0:
				places=objs[i][0]
				rotates=objs[i][1]
				sizes=objs[i][2]
			else:
				places=np.concatenate((places,objs[i][0]))
				rotates=np.concatenate((rotates,objs[i][1]))
				sizes=np.concatenate((sizes,objs[i][2]))
		
		objs_corners=get_boxcorners(places,rotates,sizes)
		ranges=[]
		for corners in objs_corners:
			corners=corners[np.lexsort((corners[:,2],corners[:,1],corners[:,0]))]
			p1=corners[0]
			p2=corners[-1]
			range_object=np.array([p1, p2]).ravel()
			_range=[range_object[0],range_object[3], range_object[1], range_object[4], range_object[2], range_object[5]]
			ranges.append(_range)
		if signal:
			object_neg=create_negatives_split(pc,ranges, objsize,size1=size1,size2=size2,emh=emh)
		else:
			object_neg=create_negatives(pc,ranges, objsize)
		if not isinstance(object_neg,type(None)):
			cropsofobject.append(object_neg)
		numberofobjects+=1
	return cropsofobject



# create negative crops
def create_negative_dataset(pths,root,dirs_crops_train,dirs_crops_test,objssizes,cars_train,peds_train,cycs_train,\
	cars_test_pos_easy,cars_test_pos_medium,cars_test_pos_hard,peds_test_pos_easy,peds_test_pos_medium,peds_test_pos_hard,\
			cycs_test_pos_easy,cycs_test_pos_medium,cycs_test_pos_hard):
	# the size of crops
	carsize=objssizes['Car_size:']
	pedsize=objssizes['Ped_size:']
	cycsize=objssizes['Cyc_size:']

	crops={'train':{'car_NEG':[],'ped_NEG':[],'cyc_NEG':[]},
	'test':{'car_NEG':{'easy':[],'medium':[],'hard':[]},
	'ped_NEG':{'easy':[],'medium':[],'hard':[]},
	'cyc_NEG':{'easy':[],'medium':[],'hard':[]}}}
	# first creating the car negatives for train cases
	neg_car_crops_train=object_crops(pths[:3],cars_train,carsize,code='0')
	crops['train']['car_NEG']=neg_car_crops_train
	# creating the ped negatives
	neg_ped_crops_train=object_crops(pths[:3],peds_train,pedsize,code='1')
	crops['train']['ped_NEG']=neg_ped_crops_train
	# creating the cyc negatives
	neg_cyc_crops_train=object_crops(pths[:3],cycs_train,cycsize,code='2')
	crops['train']['cyc_NEG']=neg_cyc_crops_train

	# first creating the car negatives for test cases
	neg_car_crops_test_easy=object_crops(pths[3:],cars_test_pos_easy,carsize,code='0',size1=150, size2=None,signal=True,emh=1)
	crops['test']['car_NEG']['easy']=neg_car_crops_test_easy
	neg_car_crops_test_medium=object_crops(pths[3:],cars_test_pos_medium,carsize,code='0',size1=50, size2=150,signal=True,emh=2)
	crops['test']['car_NEG']['medium']=neg_car_crops_test_medium
	neg_car_crops_test_hard=object_crops(pths[3:],cars_test_pos_hard,carsize,code='0',size1=10, size2=50,signal=True,emh=3)
	crops['test']['car_NEG']['hard']=neg_car_crops_test_hard
	# creating the ped negatives
	neg_ped_crops_test_easy=object_crops(pths[3:],peds_test_pos_easy,pedsize,code='1',size1=150, size2=None,signal=True,emh=1)
	crops['test']['ped_NEG']['easy']=neg_ped_crops_test_easy
	neg_ped_crops_test_medium=object_crops(pths[3:],peds_test_pos_medium,pedsize,code='1',size1=50, size2=150,signal=True,emh=2)
	crops['test']['ped_NEG']['medium']=neg_ped_crops_test_medium
	neg_ped_crops_test_hard=object_crops(pths[3:],peds_test_pos_hard,pedsize,code='1',size1=10, size2=50,signal=True,emh=3)
	crops['test']['ped_NEG']['hard']=neg_ped_crops_test_hard
	# creating the cyc negatives
	neg_cyc_crops_test_easy=object_crops(pths[3:],cycs_test_pos_easy,cycsize,code='2',size1=150, size2=None,signal=True,emh=1)
	crops['test']['cyc_NEG']['easy']=neg_cyc_crops_test_easy
	neg_cyc_crops_test_medium=object_crops(pths[3:],cycs_test_pos_medium,cycsize,code='2',size1=50, size2=150,signal=True,emh=2)
	crops['test']['cyc_NEG']['medium']=neg_cyc_crops_test_medium
	neg_cyc_crops_test_hard=object_crops(pths[3:],cycs_test_pos_hard,cycsize,code='2',size1=10, size2=50,signal=True,emh=1)
	crops['test']['cyc_NEG']['hard']=neg_cyc_crops_test_hard
	
	orig_dirs_train=dirs_create(root,dirs_crops_train)
	orig_dirs_test=dirs_create(root,dirs_crops_test)
	keys1=['train']
	keys2=['test']
	keys3=['car_NEG','ped_NEG', 'cyc_NEG']
	keys4=['easy','medium','hard']
	crops_to_file_train(crops,orig_dirs_train,keys1,keys3,signal='negative')
	crops_to_file_test(crops,orig_dirs_test,keys2,keys3,keys4,signal='negative')
	


	