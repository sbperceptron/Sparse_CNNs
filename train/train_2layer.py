import numpy as np 
import os, shutil
import multiprocessing as mp
import time,glob,random
from multiprocessing import Process
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore")
from CNN_BB import Batch_Generator,Feature_extractor
from CNN_BB import Read_Calib_File,Read_label_data
from CNN_BB import HNM
from models import nn_model2

class Train_Model2(object):
	def __init__(self, batchsize,full_pcs_bin_paths, full_pcs_labels_paths,full_pcs_calibs_paths,car_positives,neg,neg_root,\
	resolution,epochs,lr,SGDmomentum,L2weightdecay,N,weights1,weights2,weights3,b1,b2,b3,RFCar,pad,curr_epoch,ch1,ch2,f1,f2,
	x,y,z,fvthresh,wdthresh,batchsizehnm,objects,weights_path,values_path):
		super(Train_Model2, self).__init__()
		self.batchsize = batchsize
		self.full_pcs_bin_paths=full_pcs_bin_paths
		self.full_pcs_labels_paths=full_pcs_labels_paths
		self.full_pcs_calibs_paths=full_pcs_calibs_paths
		self.car_positives=car_positives
		self.neg=neg
		self.neg_root=neg_root
		self.resolution=resolution
		self.epochs=epochs
		self.lr=lr
		self.SGDmomentum=SGDmomentum
		self.L2weightdecay=L2weightdecay
		self.N=N
		self.weights1=weights1
		self.weights2=weights2
		self.weights3=weights3
		self.b1=b1
		self.b2=b2
		self.b3=b3
		self.RFCar=RFCar
		self.pad=pad
		self.curr_epoch=curr_epoch
		self.ch1=ch1
		self.ch2=ch2
		self.f1=f1
		self.f2=f2
		self.x=x
		self.y=y
		self.z=z
		self.fvthresh=fvthresh
		self.wdthresh=wdthresh
		self.batchsizehnm=batchsizehnm
		self.objects=objects
		self.weights_path=weights_path
		self.values_path=values_path

	'''Training the Neural network on the positive, negative crops and Full point clouds data'''
	def train(self):
		batchsize=self.batchsize
		full_pcs_bin_paths=self.full_pcs_bin_paths
		full_pcs_labels_paths=self.full_pcs_labels_paths
		full_pcs_calibs_paths=self.full_pcs_calibs_paths
		car_positives=self.car_positives
		neg=self.neg 
		resolution=self.resolution
		epochs=self.epochs
		lr=self.lr
		SGDmomentum=self.SGDmomentum
		L2weightdecay=self.L2weightdecay
		N=self.N
		weights1=self.weights1
		weights2=self.weights2
		weights3=self.weights3
		b1=self.b1
		b2=self.b2
		b3=self.b3
		RFCar=self.RFCar
		pad=self.pad
		curr_epoch=self.curr_epoch
		ch1=self.ch1
		ch2=self.ch2
		f1=self.f1
		f2=self.f2
		x=self.x
		y=self.y
		z=self.z
		fvthresh=self.fvthresh
		wdthresh=self.wdthresh
		batchsizehnm=self.batchsizehnm
		FPcount=0
		num_batches=0
		telapsed=0
		previous_w1_grad=0
		previous_w2_grad=0
		previous_w3_grad=0
		previous_b1_grad=0
		previous_b2_grad=0
		previous_b3_grad=0

		# remove the old hard negative mining directory
		if os.path.exists(self.neg_root):
			shutil.rmtree(self.neg_root)

		for epoch in range(curr_epoch, epochs):

			# hard negative mining
			if (epoch+1) %10 == 0 and epoch != 0:
				# remove the old hnm directory
				if os.path.exists(self.neg_root):
					shutil.rmtree(self.neg_root)
				# create the new hard negative mining folder
				os.makedirs(self.neg_root)

				hnm_obj=HNM(weights1,weights2,weights3,b1,b2,b3,full_pcs_bin_paths, 
				full_pcs_labels_paths,full_pcs_calibs_paths,self.neg_root,resolution,
				RFCar,x,y,z,fvthresh,wdthresh,batchsizehnm,self.objects)
				FPcount=hnm_obj.hnm()
				

			# Read in the positive and negatives
			# positive and negative pcs
			pos_velodynes_path=car_positives
			neg_velodynes_path =glob.glob(neg)
			pos_velodynes_path = glob.glob(pos_velodynes_path)
			pos_velodynes_path.sort()
			neg_velodynes_path.sort()
			pos_velodynes_path=pos_velodynes_path
			neg_velodynes_path=neg_velodynes_path

			# positive and negative labels
			pos_labels=[1 for i in range(0,len(pos_velodynes_path))]
			neg_labels=[-1 for i in range(0,len(neg_velodynes_path))]

			hnm_files=[]
			hnm_files_labels=[]
			# add the hnm files if they exist
			if os.path.exists(self.neg_root):
				hnm_files=glob.glob(self.neg_root+"*.bin")
				hnm_files.sort()
				hnm_files_labels=[-1 for i in range(0,len(hnm_files))]

			velodynes=pos_velodynes_path+neg_velodynes_path+hnm_files
			labels_orig=pos_labels+neg_labels+hnm_files_labels
			batch_obj=Batch_Generator(velodynes,labels_orig,RFCar,batchsize,self.fvthresh,self.wdthresh)

			batch_counter=0
			# check if there are any hard negative mining False Positives generated
			for fvs_batch,labels_batch,start_time in batch_obj.batch_generator(): 
				batch_counter+=1
				print("Epoch",epoch)
				print("Batch",batch_counter)
				feature_extractor__end_time=time.time()
				work=[]
				for i,j in zip(fvs_batch,labels_batch):
					s=False
					work.append([i,j,weights1,weights2,weights3,b1,b2,b3,RFCar,s,ch1,ch2,f1,f2])
					# nn_model2([i,j,weights1,weights2,weights3,b1,b2,b3,RFCar,s,ch1,ch2,f1,f2])
				if len(work)==0:
					continue
				# calls the batch fed function in batch generator
				################################################################################
					# Passing the input to neural network model
					# forward propogation and backward propogation 16 times	
				##################################################################################
				# USing POOL
				##################################################################################
				pool=mp.Pool(processes=8)
				results=pool.map(nn_model2,work) # nn function in the begining of file
				pool.close()
				pool.join()
				##################################################################################
				
				results=np.array(results)
				grads_1=sum(results[:,0])/(results.shape[0])
				grads_2=sum(results[:,1] )/(results.shape[0])
				grads_3=sum(results[:,2] )/(results.shape[0])
				b_grads_1= sum(results[:,3])/(results.shape[0])
				b_grads_2= sum(results[:,4])/(results.shape[0])
				b_grads_3= sum(results[:,5])/(results.shape[0])
				total_error=sum(results[:,6])/(results.shape[0])
				AVG_fwd_time=sum(results[:,8])/(results.shape[0])
				AVG_bckwd_time=sum(results[:,9])/(results.shape[0])
				
				f= open(self.values_path,"a+")
				for score,error in zip(results[:,7],results[:,6]):
					f.write("%2f %2f\r\n" % (score, error))

				# print sum(results[:,4])
				print('        Average_error: ',np.round(total_error,2))
				print('        Average time for feature_extraction:',np.round((feature_extractor__end_time-start_time),2))
				print('        Average time for forward propogation:',np.round(AVG_fwd_time,2))
				print('        Average time for Backward propogation:',np.round(AVG_bckwd_time,2))

				###############################################################################
				# Straight forward
				#######################################################################################
				# for single_w in work:
				# 	grads_1,grads_2,grad_3,b_grads_1,b_grads_2,b_grads_3,total_error,fwd_time,bckwd_time=nn_model(single_w)


				# updating the weights
				# happens for every batch (size of 16)
				################################################################################
				##############              weights1            ################################
				print("        Previous weights Average (W1):",np.average(weights1))
				
				sgd_mom_w1=((lr * (grads_1+(2*L2weightdecay*weights1)))+(SGDmomentum*previous_w1_grad))
				weights1 = weights1 - sgd_mom_w1
				print("        Updated Weights Average (W1):",np.average(weights1))
				################################################################################
				##############              weights2            ################################
				print("        Previous weights Average (W2):",np.average(weights2))
				sgd_mom_w2=((lr * (grads_2+(2*L2weightdecay*weights2)))+(SGDmomentum*previous_w2_grad))
				weights2 = weights2 - sgd_mom_w2
				print("        Updated Weights Average (W2):",np.average(weights2))
				################################################################################
				##############              weights3            ################################
				sgd_mom_w3=((lr * (grads_3.T+ (2*L2weightdecay*weights3)))+(SGDmomentum*previous_w3_grad))
				weights3 = weights3 - sgd_mom_w3 
				################################################################################
				##############              bias1            ###################################
				sgd_mom_b1=((lr * (b_grads_1+(2*L2weightdecay*b1)))+(SGDmomentum*previous_b1_grad))
				b1= b1 - sgd_mom_b1
				################################################################################
				##############              bias2            ###################################
				sgd_mom_b2=((lr * (b_grads_2+(2*L2weightdecay*b2)))+(SGDmomentum*previous_b2_grad))
				b2= b2 - sgd_mom_b2
				################################################################################
				##############              bias3            ###################################
				sgd_mom_b3=((lr * (b_grads_3+(2*L2weightdecay*b3)))+(SGDmomentum*previous_b3_grad))
				b3= b3 - sgd_mom_b3

				previous_w1_grad=sgd_mom_w1
				previous_w2_grad=sgd_mom_w2
				previous_w3_grad=sgd_mom_w3
				previous_b1_grad=sgd_mom_b1
				previous_b2_grad=sgd_mom_b2
				previous_b3_grad=sgd_mom_b3

				# making sure the bias values are non positive
				for i,b_1 in enumerate(b1):
					if b_1 > 0:
						b1[i]=0

				for j,b_2 in enumerate(b2):
					if b_2 > 0:
						b2[j]=0

				if b3 > 0:
					b3=0

				end_time=time.time()
				print("         Time elapsed for processing a Batch:",np.round((end_time-start_time),2))
				error=np.round((sum(results[:,7])/(batchsize)),2)
				telapsed+=np.round((end_time-start_time),2)


			if (epoch+1) %1 == 0 and epoch != 0:
				name=self.weights_path+'epoch_'+str(epoch+1)+'.weights'
				np.savez(name,w1=weights1,w2=weights2,w3=weights3,b1=b1,b2=b2,b3=b3)
				print(("weights saved to file", name))
			
		


		
