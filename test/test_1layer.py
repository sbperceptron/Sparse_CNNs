import numpy as np 
import os, shutil
import multiprocessing as mp
import time,glob,random
from multiprocessing import Process
import torch
import torch.nn as nn
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore")
from CNN_BB import Batch_Generator,Feature_extractor
from CNN_BB import Read_Calib_File,Read_label_data
from CNN_BB import HNM
from models import nn_model2
from evaluation import PR_Curves,ROC_Curves

class Test_Model1(object):
	def __init__(self,batchsize,objects_pos_easy_pcd,
		objects_pos_medium_pcd,objects_pos_hard_pcd,objects_neg_easy_pcd,
		objects_neg_medium_pcd,objects_neg_hard_pcd,weights1,
		weights2,weights3,b1,b2,b3,RFCar,ch1,ch2,f1,f2,
		curr_epoch,dst_path,fvthresh):
		self.batchsize=batchsize
		self.objects_pos_easy_pcd=objects_pos_easy_pcd
		self.objects_pos_medium_pcd=objects_pos_medium_pcd
		self.objects_pos_hard_pcd=objects_pos_hard_pcd
		self.objects_neg_easy_pcd=objects_neg_easy_pcd
		self.objects_neg_medium_pcd=objects_neg_medium_pcd
		self.objects_neg_hard_pcd=objects_neg_hard_pcd
		self.weights1=weights1
		self.weights2=weights2
		self.weights3=weights3
		self.b1=b1
		self.b2=b2
		self.b3=b3
		self.RFCar=RFCar
		self.ch1=ch1
		self.ch2=ch2
		self.f1=f1
		self.f2=f2
		self.curr_epoch=curr_epoch
		self.dst_path=dst_path
		self.fvthresh=fvthresh


	# Read in the positive and negatives
	# positive and negative pcs
	def test(self):
		test_pos_easy_path = self.objects_pos_easy_pcd
		test_pos_medium_path = self.objects_pos_medium_pcd
		test_pos_hard_path = self.objects_pos_hard_pcd

		test_neg_easy_path = self.objects_neg_easy_pcd
		test_neg_medium_path = self.objects_neg_medium_pcd
		test_neg_hard_path = self.objects_neg_hard_pcd

		test_pos_easy_path = glob.glob(test_pos_easy_path)
		test_pos_medium_path = glob.glob(test_pos_medium_path)
		test_pos_hard_path = glob.glob(test_pos_hard_path)

		test_neg_easy_path = glob.glob(test_neg_easy_path)
		test_neg_medium_path = glob.glob(test_neg_medium_path)
		test_neg_hard_path = glob.glob(test_neg_hard_path)

		# print len(test_pos_easy_path),len(test_pos_medium_path),len(test_pos_hard_path),\
		# len(test_neg_easy_path),len(test_neg_medium_path),len(test_neg_hard_path)

		test_pos_easy_path.sort()
		test_pos_medium_path.sort()
		test_pos_hard_path.sort()

		test_neg_easy_path.sort()
		test_neg_medium_path.sort()
		test_neg_hard_path.sort()

		test_pos_easy_path=test_pos_easy_path
		test_pos_medium_path=test_pos_medium_path
		test_pos_hard_path=test_pos_hard_path

		test_neg_easy_path=test_neg_easy_path
		test_neg_medium_path=test_neg_medium_path
		test_neg_hard_path=test_neg_hard_path

		# print len(test_pos_easy_path),len(test_pos_medium_path),len(test_pos_hard_path),\
		# len(test_neg_easy_path),len(test_neg_medium_path),len(test_neg_hard_path)

		pos_velodynes_easy=test_pos_easy_path
		pos_velodynes_medium=test_pos_medium_path
		pos_velodynes_hard=test_pos_hard_path

		neg_velodynes_easy=test_neg_easy_path
		neg_velodynes_medium=test_neg_medium_path
		neg_velodynes_hard=test_neg_hard_path

		pos_labels_easy=[1 for i in range(0,len(pos_velodynes_easy))]
		pos_labels_medium=[1 for i in range(0,len(pos_velodynes_medium))]
		pos_labels_hard=[1 for i in range(0,len(pos_velodynes_hard))]

		neg_labels_easy=[-1 for i in range(0,len(neg_velodynes_easy))]
		neg_labels_medium=[-1 for i in range(0,len(neg_velodynes_medium))]
		neg_labels_hard=[-1 for i in range(0,len(neg_velodynes_hard))]

		velodynes_easy=pos_velodynes_easy+neg_velodynes_easy
		velodynes_medium=pos_velodynes_medium+neg_velodynes_medium
		velodynes_hard=pos_velodynes_hard+neg_velodynes_hard

		labels_easy=pos_labels_easy+neg_labels_easy
		labels_medium=pos_labels_medium+neg_labels_medium
		labels_hard=pos_labels_hard+neg_labels_hard


		allpreds=[]
		allground=[]
		for velodynes,labels in zip([velodynes_easy,velodynes_medium,velodynes_hard],[labels_easy,labels_medium,labels_hard]):
			
			total_number_batches=len(velodynes)//self.batchsize
			curr_batch=1
			predictions=[]
			groundtruth=[]
			batch_obj=Batch_Generator(velodynes,labels,self.RFCar,self.batchsize,self.fvthresh)
			for fvs_batch,labels_batch,start_time in batch_obj.batch_generator(): 
				print "Processing Batch "+str(curr_batch)+" of "+str(total_number_batches)
				curr_batch+=1
				feature_extractor__end_time = time.time()
				work=[]
				for i,j in zip(fvs_batch,labels_batch):
					s=True
					# single layer model
					# work.append([i,j,weights1,weights2,b1,b2,RFCar,s,ch1,f1])
					# two layer model
					work.append([i,j,self.weights1,self.weights2,self.weights3,self.b1,
						self.b2,self.b3,self.RFCar,s,self.ch1,self.ch2,self.f1,self.f2])
					
				# for single layer replace the nn_2_layer_model with nn_model
				pool=mp.Pool(processes=8)
				results1=pool.map(nn_model2,work) # nn function in the begining of file
				pool.close()
				pool.join()
				
				results1=np.array(results1)
				total_loss=sum(results1[:,0])/(results1.shape[0])
				output_1=results1[:,1]
				for i in output_1:
					predictions.append(i)
					
				# predictions.append(results[:,6])
				groundtruth.append(labels_batch)
				
				end_time=time.time()
				print "\t Average loss value", np.round((total_loss),2)
				print "\t Time elapsed for processing a Batch:",np.round((end_time-start_time),2)

			groundtruth=[k2 for subl2 in groundtruth for k2 in subl2]
			# # predictions=[k1 for subl1 in predictions for k1 in subl1]
			# print len(predictions),len(groundtruth)
			allpreds.append(predictions)
			allground.append(groundtruth)

		pr=PR_Curves(allpreds,allground,curr_epoch,dst_path)
		pr.pr_curves()
		roc=ROC_Curves(allpreds,allground,curr_epoch,dst_path)
		roc.roc_curves()

