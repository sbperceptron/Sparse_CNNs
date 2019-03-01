from help_functions import *
# from pc_publish import *
from create_positive import *
from create_negative import *
from objs_receptive_field import *
from hard_medium_easy_split import *

'''Creates positive and negative crops from the dataset'''
def create_neg_pos(root,train_lidars,train_labels,train_calibs,test_lidars,test_labels,test_calibs):

	# For each object the 95 perecentile size is returened when we call this function
	# refer to the function definition in the objs_receptive_field.py for detailed understanding
	# refer to section 3 fourth paragraph of the paper for details
	objsizes=find_object_sizes(train_labels,train_calibs,root)

	# check if there exists the positive crops directory 
	if not os.path.exists(root+"/train/positive"):
		
		dirs_crops=["/train/positive/Car","/train/positive/Pedestrian","/train/positive/Cyclist","/test/positive/Car", \
		"/test/positive/Pedestrian","/test/positive/Cyclist"]
		dirs_labels=["/train/positive/Car_labels","/train/positive/Ped_labels","/train/positive/Cyc_labels"]
		pths2=list([train_lidars,  train_labels, train_calibs, test_lidars, test_labels, test_calibs])
		
		# Creates the Ground truth and the Positive training Crops Datsets
		create_positive_dataset(pths2,root,dirs_crops,objsizes)
		car_crops=glob.glob(root+"/train/positive/Car/*.bin")
		ped_crops=glob.glob(root+"/train/positive/Pedestrian/*.bin")
		cyc_crops=glob.glob(root+"/train/positive/Cyclist/*.bin")
		car_test_crops_pos=glob.glob(root+"/test/positive/Car/*.bin")
		ped_test_crops_pos=glob.glob(root+"/test/positive/Pedestrian/*.bin")
		cyc_test_crops_pos=glob.glob(root+"/test/positive/Cyclist/*.bin")
		car_test_crops_pos=glob.glob(root+"/test/positive/Car/*.bin")
		ped_test_crops_pos=glob.glob(root+"/test/positive/Pedestrian/*.bin")
		cyc_test_crops_pos=glob.glob(root+"/test/positive/Cyclist/*.bin")
		cars_train,peds_train,cycs_train = len(car_crops),len(ped_crops),len(cyc_crops)
		cars_test_pos,peds_test_pos,cycs_test_pos= len(car_test_crops_pos),len(ped_test_crops_pos),len(cyc_test_crops_pos)
		# print "number of deleted crops:",zeroloc
		print "\n"
		print "Train"
		print "    Positives"
		print "Car" ,cars_train
		print "Ped" ,peds_train
		print "Cyc" ,cycs_train
		print "\n"
		print "Test"
		print "    Positives"
		print "Car" ,cars_test_pos
		print "Ped" ,peds_test_pos
		print "Cyc" ,cycs_test_pos
		print "\n"
		# for creating the positive crops dataset the above function which forms the part of the create_positive.py file is called
		# refer to create_positive.py for details
	else:
		car_crops=glob.glob(root+"/train/positive/Car/*.bin")
		ped_crops=glob.glob(root+"/train/positive/Pedestrian/*.bin")
		cyc_crops=glob.glob(root+"/train/positive/Cyclist/*.bin")
		car_test_crops_pos=glob.glob(root+"/test/positive/Car/*.bin")
		ped_test_crops_pos=glob.glob(root+"/test/positive/Pedestrian/*.bin")
		cyc_test_crops_pos=glob.glob(root+"/test/positive/Cyclist/*.bin")
		cars_train,peds_train,cycs_train = len(car_crops),len(ped_crops),len(cyc_crops)
		cars_test_pos,peds_test_pos,cycs_test_pos= len(car_test_crops_pos),len(ped_test_crops_pos),len(cyc_test_crops_pos)
		# print "number of deleted crops:",zeroloc
		print "\n"
		print "Train"
		print "    Positives"
		print "Car" ,cars_train
		print "Ped" ,peds_train
		print "Cyc" ,cycs_train
		print "\n"
		print "Test"
		print "    Positives"
		print "Car" ,cars_test_pos
		print "Ped" ,peds_test_pos
		print "Cyc" ,cycs_test_pos
		print "\n"

	#check if the test dataset with easy,moderate and hard split is created.
	if not os.path.exists(root+"/test/split/positive"):
		dirs_easy_medium_hard_crops_pos=["/test/split/positive/Car/easy", "/test/split/positive/Car/medium", "/test/split/positive/Car/hard",\
		"/test/split/positive/Pedestrian/easy", "/test/split/positive/Pedestrian/medium", "/test/split/positive/Pedestrian/hard",
		"/test/split/positive/Cyclist/easy", "/test/split/positive/Cyclist/medium", "/test/split/positive/Cyclist/hard"]
		pths4=list([car_test_crops_pos, ped_test_crops_pos, cyc_test_crops_pos])
		
		# Creates the Negative training Crops Datset
		create_easy_medium_hard_split_pos(pths4,root,dirs_easy_medium_hard_crops_pos)

		car_crops_pos_test_easy=glob.glob(root+"/test/split/positive/Car/easy/*.bin")
		car_crops_pos_test_medium=glob.glob(root+"/test/split/positive/Car/medium/*.bin")
		car_crops_pos_test_hard=glob.glob(root+"/test/split/positive/Car/hard/*.bin")
		
		ped_crops_pos_test_easy=glob.glob(root+"/test/split/positive/Pedestrian/easy/*.bin")
		ped_crops_pos_test_medium=glob.glob(root+"/test/split/positive/Pedestrian/medium/*.bin")
		ped_crops_pos_test_hard=glob.glob(root+"/test/split/positive/Pedestrian/hard/*.bin")
		
		cyc_crops_pos_test_easy=glob.glob(root+"/test/split/positive/Cyclist/easy/*.bin")
		cyc_crops_pos_test_medium=glob.glob(root+"/test/split/positive/Cyclist/medium/*.bin")
		cyc_crops_pos_test_hard=glob.glob(root+"/test/split/positive/Cyclist/hard/*.bin")

		cars_test_pos_easy,cars_test_pos_medium,cars_test_pos_hard= len(car_crops_pos_test_easy),len(car_crops_pos_test_medium),len(car_crops_pos_test_hard)
		peds_test_pos_easy,peds_test_pos_medium,peds_test_pos_hard= len(ped_crops_pos_test_easy),len(ped_crops_pos_test_medium),len(ped_crops_pos_test_hard)
		cycs_test_pos_easy,cycs_test_pos_medium,cycs_test_pos_hard= len(cyc_crops_pos_test_easy),len(cyc_crops_pos_test_medium),len(cyc_crops_pos_test_hard)
		print "\n"
		print "Test"
		print "Postives split:"
		print "Car "+str(cars_test_pos_easy)+" "+str(cars_test_pos_medium)+" "+str(cars_test_pos_hard)
		print "Ped "+str(peds_test_pos_easy)+" "+str(peds_test_pos_medium)+" "+str(peds_test_pos_hard)
		print "Cyc "+str(cycs_test_pos_easy)+" "+str(cycs_test_pos_medium)+" "+str(cycs_test_pos_hard)
		print "\n"


	else:
		car_crops_pos_test_easy=glob.glob(root+"/test/split/positive/Car/easy/*.bin")
		car_crops_pos_test_medium=glob.glob(root+"/test/split/positive/Car/medium/*.bin")
		car_crops_pos_test_hard=glob.glob(root+"/test/split/positive/Car/hard/*.bin")
		
		ped_crops_pos_test_easy=glob.glob(root+"/test/split/positive/Pedestrian/easy/*.bin")
		ped_crops_pos_test_medium=glob.glob(root+"/test/split/positive/Pedestrian/medium/*.bin")
		ped_crops_pos_test_hard=glob.glob(root+"/test/split/positive/Pedestrian/hard/*.bin")
		
		cyc_crops_pos_test_easy=glob.glob(root+"/test/split/positive/Cyclist/easy/*.bin")
		cyc_crops_pos_test_medium=glob.glob(root+"/test/split/positive/Cyclist/medium/*.bin")
		cyc_crops_pos_test_hard=glob.glob(root+"/test/split/positive/Cyclist/hard/*.bin")


		cars_test_pos_easy,cars_test_pos_medium,cars_test_pos_hard= len(car_crops_pos_test_easy),len(car_crops_pos_test_medium),len(car_crops_pos_test_hard)
		peds_test_pos_easy,peds_test_pos_medium,peds_test_pos_hard= len(ped_crops_pos_test_easy),len(ped_crops_pos_test_medium),len(ped_crops_pos_test_hard)
		cycs_test_pos_easy,cycs_test_pos_medium,cycs_test_pos_hard= len(cyc_crops_pos_test_easy),len(cyc_crops_pos_test_medium),len(cyc_crops_pos_test_hard)
		print "\n"
		print "Test"
		print "Postives split:"
		print "Car "+str(cars_test_pos_easy)+" "+str(cars_test_pos_medium)+" "+str(cars_test_pos_hard)
		print "Ped "+str(peds_test_pos_easy)+" "+str(peds_test_pos_medium)+" "+str(peds_test_pos_hard)
		print "Cyc "+str(cycs_test_pos_easy)+" "+str(cycs_test_pos_medium)+" "+str(cycs_test_pos_hard)
		print "\n"

	# check if there exists the negative crops directory 
	if not os.path.exists(root+"/train/negative"):
		dirs_neg_crops_train=["/train/negative/Car", "/train/negative/Pedestrian", "/train/negative/Cyclist"]
		dirs_neg_crops_test=["/test/split/negative/Car/easy","/test/split/negative/Car/medium","/test/split/negative/Car/hard",
		 "/test/split/negative/Pedestrian/easy","/test/split/negative/Pedestrian/medium","/test/split/negative/Pedestrian/hard"
		 ,"/test/split/negative/Cyclist/easy","/test/split/negative/Cyclist/medium","/test/split/negative/Cyclist/hard"]
		pths3=list([train_lidars,  train_labels, train_calibs,test_lidars, test_labels, test_calibs])
		
		# Creates the Negative training Crops Datset
		create_negative_dataset(pths3,root,dirs_neg_crops_train,dirs_neg_crops_test,objsizes,cars_train,peds_train,cycs_train,\
			cars_test_pos_easy,cars_test_pos_medium,cars_test_pos_hard,peds_test_pos_easy,peds_test_pos_medium,peds_test_pos_hard,\
			cycs_test_pos_easy,cycs_test_pos_medium,cycs_test_pos_hard)
		# for creating the negative crops dataset the above function which forms the part of the create_negative.py file is called
		# refer to create_negative.py for details
		car_crops_neg_train=glob.glob(root+"/train/negative/Car/*.bin")
		ped_crops_neg_train=glob.glob(root+"/train/negative/Pedestrian/*.bin")
		cyc_crops_neg_train=glob.glob(root+"/train/negative/Cyclist/*.bin")
		
		car_crops_neg_test_easy=glob.glob(root+"/test/split/negative/Car/easy/*.bin")
		car_crops_neg_test_medium=glob.glob(root+"/test/split/negative/Car/medium/*.bin")
		car_crops_neg_test_hard=glob.glob(root+"/test/split/negative/Car/hard/*.bin")
		
		ped_crops_neg_test_easy=glob.glob(root+"/test/split/negative/Pedestrian/easy/*.bin")
		ped_crops_neg_test_medium=glob.glob(root+"/test/split/negative/Pedestrian/medium/*.bin")
		ped_crops_neg_test_hard=glob.glob(root+"/test/split/negative/Pedestrian/hard/*.bin")
		
		cyc_crops_neg_test_easy=glob.glob(root+"/test/split/negative/Cyclist/easy/*.bin")
		cyc_crops_neg_test_medium=glob.glob(root+"/test/split/negative/Cyclist/medium/*.bin")
		cyc_crops_neg_test_hard=glob.glob(root+"/test/split/negative/Cyclist/hard/*.bin")

		cars_train,peds_train,cycs_train= len(car_crops_neg_train),len(ped_crops_neg_train),len(cyc_crops_neg_train)
		cars_test_neg_easy,cars_test_neg_medium,cars_test_neg_hard= len(car_crops_neg_test_easy),len(car_crops_neg_test_medium),len(car_crops_neg_test_hard)
		peds_test_neg_easy,peds_test_neg_medium,peds_test_neg_hard= len(ped_crops_neg_test_easy),len(ped_crops_neg_test_medium),len(ped_crops_neg_test_hard)
		cycs_test_neg_easy,cycs_test_neg_medium,cycs_test_neg_hard= len(cyc_crops_neg_test_easy),len(cyc_crops_neg_test_medium),len(cyc_crops_neg_test_hard)
		# print "number of deleted crops:",zeroloc
		print "\n"
		print "Train"
		print "    Negatives"
		print "Car" ,cars_train
		print "Ped" ,peds_train
		print "Cyc" ,cycs_train
		print "\n"
		print "Test"
		print "Negatives and the split"
		print "Car "+str(cars_test_neg_easy)+" "+str(cars_test_neg_medium)+" "+str(cars_test_neg_hard)
		print "Ped "+str(peds_test_neg_easy)+" "+str(peds_test_neg_medium)+" "+str(peds_test_neg_hard)
		print "Cyc "+str(cycs_test_neg_easy)+" "+str(cycs_test_neg_medium)+" "+str(cycs_test_neg_hard)
		print "\n"
	
	else:
		car_crops_neg_train=glob.glob(root+"/train/negative/Car/*.bin")
		ped_crops_neg_train=glob.glob(root+"/train/negative/Pedestrian/*.bin")
		cyc_crops_neg_train=glob.glob(root+"/train/negative/Cyclist/*.bin")
		
		car_crops_neg_test_easy=glob.glob(root+"/test/split/negative/Car/easy/*.bin")
		car_crops_neg_test_medium=glob.glob(root+"/test/split/negative/Car/medium/*.bin")
		car_crops_neg_test_hard=glob.glob(root+"/test/split/negative/Car/hard/*.bin")
		
		ped_crops_neg_test_easy=glob.glob(root+"/test/split/negative/Pedestrian/easy/*.bin")
		ped_crops_neg_test_medium=glob.glob(root+"/test/split/negative/Pedestrian/medium/*.bin")
		ped_crops_neg_test_hard=glob.glob(root+"/test/split/negative/Pedestrian/hard/*.bin")
		
		cyc_crops_neg_test_easy=glob.glob(root+"/test/split/negative/Cyclist/easy/*.bin")
		cyc_crops_neg_test_medium=glob.glob(root+"/test/split/negative/Cyclist/medium/*.bin")
		cyc_crops_neg_test_hard=glob.glob(root+"/test/split/negative/Cyclist/hard/*.bin")

		cars_train,peds_train,cycs_train= len(car_crops_neg_train),len(ped_crops_neg_train),len(cyc_crops_neg_train)
		cars_test_neg_easy,cars_test_neg_medium,cars_test_neg_hard= len(car_crops_neg_test_easy),len(car_crops_neg_test_medium),len(car_crops_neg_test_hard)
		peds_test_neg_easy,peds_test_neg_medium,peds_test_neg_hard= len(ped_crops_neg_test_easy),len(ped_crops_neg_test_medium),len(ped_crops_neg_test_hard)
		cycs_test_neg_easy,cycs_test_neg_medium,cycs_test_neg_hard= len(cyc_crops_neg_test_easy),len(cyc_crops_neg_test_medium),len(cyc_crops_neg_test_hard)
		# print "number of deleted crops:",zeroloc
		print "\n"
		print "Train"
		print "    Negatives"
		print "Car" ,cars_train
		print "Ped" ,peds_train
		print "Cyc" ,cycs_train
		print "\n"
		print "Test"
		print "Negatives and the split:"
		print "Car "+str(cars_test_neg_easy)+" "+str(cars_test_neg_medium)+" "+str(cars_test_neg_hard)
		print "Ped "+str(peds_test_neg_easy)+" "+str(peds_test_neg_medium)+" "+str(peds_test_neg_hard)
		print "Cyc "+str(cycs_test_neg_easy)+" "+str(cycs_test_neg_medium)+" "+str(cycs_test_neg_hard)
		print "\n"
