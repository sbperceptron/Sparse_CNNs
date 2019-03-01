from help_functions import *
from createnegposfn import *

# redundant
def receptive_size(objsize):
	sz=np.amax(objsize)
	if sz%2==1:
		obj_sz=[sz,sz,sz]
	else:
		obj_sz=[sz+1,sz+1,sz+1]
	return obj_sz

def getobjssizes(final_label):
	if not final_label==None:
		if str(0) in final_label.keys():
			cars=final_label[str(0)][2]
		else:
			cars=None
		if str(1) in final_label.keys():
			peds=final_label[str(1)][2]
		else:
			peds=None
		if str(2) in final_label.keys():
			cycs=final_label[str(2)][2]
		else:
			cycs=None
		objectsizes={'carsizes':cars,'pedsizes':peds,'cycsizes':cycs}
		return objectsizes
	else:
		return None

'''loop over all the object label files and return the 95 perecentile size'''
def find_object_sizes(labels, calibs,root):
	carssizes=[]
	pedssizes=[]
	cycssizes=[]
	count=0
	for i,j in zip(labels,calibs):
		count +=1
		label_type="txt"
		calib = read_calib_file(j)
		proj_velo = proj_to_velo(calib)[:, :3]
		label_type="txt"
		
		final_label= read_labels(i, label_type, calib_path=calib, is_velo_cam=True, proj_velo=proj_velo)
		objssizes=getobjssizes(final_label)
		# print type(objssizes['carsizes'])
		if objssizes:
			# isinstance(None, type(None))
			if not isinstance(objssizes['carsizes'], type(None)): 
				carsizes=objssizes['carsizes']
				carssizes.append(carsizes)

			if not isinstance(objssizes['pedsizes'] ,type(None)): 
				pedsizes=objssizes['pedsizes']
				pedssizes.append(pedsizes)

			if not isinstance(objssizes['cycsizes'] ,type(None)): 
				cycsizes=objssizes['cycsizes']
				cycssizes.append(cycsizes)

	# All the collected objects sizes are used to caliculate the object size
	########################################################################
	all_cars=[k.tolist() for sublist in carssizes for k in sublist]
	carsize = np.percentile(np.array(all_cars), 95, axis=0)

	all_peds=[k for sublist in pedssizes for k in sublist]
	all_peds_ar=np.array(all_peds)

	pedsize = np.percentile(all_peds_ar, 95, axis=0)
	all_cycs=[k for sublist in cycssizes for k in sublist]
	all_cycs_ar=np.array(all_cycs)
	cycsize = np.percentile(all_cycs_ar, 95, axis=0)
	############################################################################

	# the original size of the object is stored into the file
	print " 95 percentile object sizes based on "+str(count)+" files"
	print "Car_size:",carsize
	print "Ped_size:",pedsize
	print "Cyc_size:",cycsize

	# the size of object is ceiled to get the closest integer value for each object
	carsize=np.ceil(carsize).astype(np.int16)
	pedsize=np.ceil(pedsize).astype(np.int16)
	cycsize=np.ceil(cycsize).astype(np.int16)


	sizelist="# HEIGHT WIDTH AND LENGTH  format\n"
	sizelist+=("Car size:"+str(carsize)+"\n" +"Pedestrian size:"+str(pedsize) +"\n"+ "Cyclist size:"+str(cycsize))
	dire=root+"/"
	name="objects_sizes.txt"
	
	np.array(sizelist).tofile(dire+name, sep= ',')

	objs=dict()
	objs['Car_size:']=carsize
	objs['Ped_size:']=pedsize
	objs['Cyc_size:']=cycsize

	print "Car_receptive_field:",carsize
	print "Ped_receptive_field:",pedsize
	print "Cyc_receptive_field:",cycsize

	return objs