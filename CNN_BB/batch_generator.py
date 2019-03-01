import numpy as np
import glob
import os
from sklearn.utils import shuffle
import time
from .feature_extractor import Feature_extractor


class Batch_Generator:
	def __init__(self,velodynes,labels,RFCar,batch_size,fvthresh,wdthresh):
		self.velodynes=velodynes
		self.labels=labels
		self.RFCar=RFCar
		self.batch_size=batch_size
		self.fvthresh=fvthresh
		self.wdthresh=wdthresh

	'''Rotate about the z axis of the object'''
	def zaxis_rotate(self,newpc,angle):
		pc=np.array(newpc)
		"""Rotate all the points about z by a fixed angle"""
		Rz=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0], [0,0,1]]
		zrotation=[]
		final_rot=np.zeros((pc.shape))
		rot_pc=pc[:,:3].dot(Rz)
		final_rot[:,:3]=rot_pc
		final_rot[:,3]=pc[:,3]
		return final_rot

	'''Feeding N number of point clouds to the training function. Where N is equal to Batch size'''
	def batch_generator(self):
		iterations=len(self.velodynes)//self.batch_size
		# shuffling the data
		velodynes,labels=shuffle(self.velodynes,self.labels,random_state=0)
		for itr in range(0,iterations):
			labels_pcs=[]
			pcs=[]
			fvs_Batch=[]
			labels_batch=[]
			grids=[]
			nonzeroloc_pcs=[]
			labels_pcs=[]
			for pc_name, label in zip(velodynes[itr*self.batch_size:(itr+1)*self.batch_size],
				labels[itr*self.batch_size:(itr+1)*self.batch_size]):
				start_time=time.time()

				fvs=Feature_extractor(pc_name,self.RFCar,dtype="bin",resolution=0.20,
					fvthresh=self.fvthresh)
				FVS,counter=fvs.feature_extractor()

				if counter<self.wdthresh:
					continue

				if FVS != None:
					# print counter
					fvs_Batch.append(FVS)
					labels_batch.append(label)
			yield fvs_Batch,labels_batch,start_time
	
	
	
		
