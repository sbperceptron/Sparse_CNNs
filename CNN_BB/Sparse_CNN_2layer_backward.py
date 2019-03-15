# import data as dt
from sklearn.metrics import hinge_loss
from scipy.signal import convolve
import time
import numpy as np
from .Activations import Activation_Functions
from .non_zerofvs import Nonzero_FVS

class Sparse_CNN_2layer_backward(object):
	"""docstring for Sparse_CNN_2layer_backward"""
	def __init__(self, cost,label,yhat,conv1,conv2,FC1,w1,b1,w2,b2,w3,b3,
		dx1_dw1,dx1_db1,dx2_dw2,dx2_db2,ch1,ch2,f1,f2,RFCar):
		super(Sparse_CNN_2layer_backward, self).__init__()
		self.cost = cost
		self.label=label
		self.yhat=yhat
		self.conv1=conv1
		self.conv2=conv2
		self.FC1=FC1
		self.w1=w1
		self.b1=b1
		self.w2=w2
		self.b2=b2
		self.w3=w3
		self.b3=b3
		self.dx1_dw1=dx1_dw1
		self.dx1_db1=dx1_db1
		self.dx2_dw2=dx2_dw2
		self.dx2_db2=dx2_db2
		self.ch1=ch1
		self.ch2=ch2
		self.f1=f1
		self.f2=f2
		self.RFCar=RFCar

		
	def dE_dout(self):
		if (self.label*self.yhat) < 1:
			return -self.label
		else:
			return 0

	def dout_dx3(self,w3):
		return w3

	def dx3_drelu2(self):
		return 1

	def drelu2_dx2(self):
		flat_conv2=np.expand_dims(self.conv2.ravel(),0)
		drelu_obj=Activation_Functions(flat_conv2)
		return drelu_obj.d_relu()

	def dx2_drelu1(self):
		return self.w2
		
	def drelu1_dx1(self):
		flat_conv1=np.expand_dims(self.conv1.ravel(),0)
		drelu_obj=Activation_Functions(flat_conv1)
		return drelu_obj.d_relu()

	def dout_dw3(self):
		return self.FC1

	def dE_dw3(self):
		gradforw3=self.dE_dout()*self.dout_dw3()
		return gradforw3

	def dE_db3(self):
		gradforb3=self.dE_dout()
		return gradforb3

	def dE_dw2(self):
		grad=self.dx2_dw2
		gradforw2_part1=(self.dE_dout()*self.dout_dx3(self.w3.T)*self.dx3_drelu2()*self.drelu2_dx2()).reshape(-1,self.f2)
		gradforw2=np.zeros((self.w2.shape))
		f2=self.f2
		for ch in range(0,self.ch2):
			idx=0
			for x in range(0,2):
				for y in range (0,2):
					for z in range (0,2):
						gradforw2[x,y,z,ch,:]=np.sum(gradforw2_part1*
							                         (grad[idx][:, :, :, ch, :].reshape(-1, f2)),
							                         axis=0)
						idx+=1
'''
			gradforw2[0,0,0,ch,:]=np.sum(gradforw2_part1*(grad[0][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[0,0,1,ch,:]=np.sum(gradforw2_part1*(grad[1][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[0,0,2,ch,:]=np.sum(gradforw2_part1*(grad[2][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[0,1,0,ch,:]=np.sum(gradforw2_part1*(grad[3][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[0,1,1,ch,:]=np.sum(gradforw2_part1*(grad[4][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[0,1,2,ch,:]=np.sum(gradforw2_part1*(grad[5][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[0,2,0,ch,:]=np.sum(gradforw2_part1*(grad[6][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[0,2,1,ch,:]=np.sum(gradforw2_part1*(grad[7][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[0,2,2,ch,:]=np.sum(gradforw2_part1*(grad[8][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[1,0,0,ch,:]=np.sum(gradforw2_part1*(grad[9][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[1,0,1,ch,:]=np.sum(gradforw2_part1*(grad[10][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[1,0,2,ch,:]=np.sum(gradforw2_part1*(grad[11][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[1,1,0,ch,:]=np.sum(gradforw2_part1*(grad[12][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[1,1,1,ch,:]=np.sum(gradforw2_part1*(grad[13][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[1,1,2,ch,:]=np.sum(gradforw2_part1*(grad[14][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[1,2,0,ch,:]=np.sum(gradforw2_part1*(grad[15][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[1,2,1,ch,:]=np.sum(gradforw2_part1*(grad[16][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[1,2,2,ch,:]=np.sum(gradforw2_part1*(grad[17][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[2,0,0,ch,:]=np.sum(gradforw2_part1*(grad[18][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[2,0,1,ch,:]=np.sum(gradforw2_part1*(grad[19][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[2,0,2,ch,:]=np.sum(gradforw2_part1*(grad[20][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[2,1,0,ch,:]=np.sum(gradforw2_part1*(grad[21][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[2,1,1,ch,:]=np.sum(gradforw2_part1*(grad[22][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[2,1,2,ch,:]=np.sum(gradforw2_part1*(grad[23][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[2,2,0,ch,:]=np.sum(gradforw2_part1*(grad[24][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[2,2,1,ch,:]=np.sum(gradforw2_part1*(grad[25][:,:,:,ch,:].reshape(-1,f2)),axis=0)
			gradforw2[2,2,2,ch,:]=np.sum(gradforw2_part1*(grad[26][:,:,:,ch,:].reshape(-1,f2)),axis=0)
'''

		return gradforw2

	def dE_db2(self):
		gradforb2_part1=self.dE_dout()*self.dout_dx3(self.w3.T)*self.dx3_drelu2()*self.drelu2_dx2()
		
		gradforb2=np.zeros((self.b2.shape))
		# TODO: change the value 8 to the number of filters
		for i in range(0,8):
			dummy=np.zeros((self.conv2.shape))
			dummy[:,:,:,i]=self.dx2_db2[:,:,:,i]
			flat_db=np.expand_dims(dummy.ravel(),0)
			gradforb2[i]=sum((gradforb2_part1*flat_db).T)
		return gradforb2

	def gradforw1_part2cal(self,gradforw1_part1,weights,f):
		RFCar=self.RFCar
		nz_obj=Nonzero_FVS(gradforw1_part1)
		FVS=nz_obj.nonzerofeatures()
		numfilters=weights.shape[-1]
		scores=np.zeros((RFCar[0],RFCar[1],RFCar[2],f))
		
		for i in range(0,weights.shape[-1]):
			score=np.zeros((RFCar[0],RFCar[1],RFCar[2]))
			weight=weights[:,:,:,:,i]
			weight=np.flip(weight,axis=0)
			weight=np.flip(weight,axis=1)
			weight=np.flip(weight,axis=2)

			for cell, FV in FVS.items():
				count=0
#				FV=FVS[cell]
				c=np.array(cell, dtype=int)
				# TODO: remove the conditional statement from here
				if (c[0]+2) < (RFCar[0]) and (c[1]+2)<RFCar[1] and (c[2]+2)<RFCar[2] and c[0]!=0 and c[1]!=0 and c[2]!=0:
					
					score[c[0]-1:c[0]+2,c[1]-1:c[1]+2,c[2]-1:c[2]+2]+=np.sum(weight*FV, axis=-1)			
			
			scores[:,:,:,i]=score

		scores=np.expand_dims(scores.ravel(),0)
		
		return scores


	def dE_dw1(self):
		cost=self.cost
		label=self.label
		yhat=self.yhat
		w3=self.w3
		conv2=self.conv2
		w2=self.w2
		conv1=self.conv1
		w1=self.w1
		ch1=self.ch1
		f1=self.f1
		RFCar=self.RFCar
		grad=self.dx1_dw1
		gradforw1_part1 = (self.dE_dout()*self.dout_dx3(self.w3.T)*self.dx3_drelu2()*
			self.drelu2_dx2()).reshape(RFCar[0],RFCar[1],RFCar[2],f1)
		gradforw1_part2 = self.gradforw1_part2cal(gradforw1_part1, self.dx2_drelu1(), f1)
		gradforw1_part3 = (gradforw1_part2 *self.drelu1_dx1()).reshape(-1,f1)
		gradforw1=np.zeros((w1.shape))
		for ch in range(0,ch1):
			gradforw1[0,0,0,ch,:]=np.sum(gradforw1_part3*(grad[0][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[0,0,1,ch,:]=np.sum(gradforw1_part3*(grad[1][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[0,0,2,ch,:]=np.sum(gradforw1_part3*(grad[2][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[0,1,0,ch,:]=np.sum(gradforw1_part3*(grad[3][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[0,1,1,ch,:]=np.sum(gradforw1_part3*(grad[4][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[0,1,2,ch,:]=np.sum(gradforw1_part3*(grad[5][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[0,2,0,ch,:]=np.sum(gradforw1_part3*(grad[6][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[0,2,1,ch,:]=np.sum(gradforw1_part3*(grad[7][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[0,2,2,ch,:]=np.sum(gradforw1_part3*(grad[8][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[1,0,0,ch,:]=np.sum(gradforw1_part3*(grad[9][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[1,0,1,ch,:]=np.sum(gradforw1_part3*(grad[10][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[1,0,2,ch,:]=np.sum(gradforw1_part3*(grad[11][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[1,1,0,ch,:]=np.sum(gradforw1_part3*(grad[12][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[1,1,1,ch,:]=np.sum(gradforw1_part3*(grad[13][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[1,1,2,ch,:]=np.sum(gradforw1_part3*(grad[14][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[1,2,0,ch,:]=np.sum(gradforw1_part3*(grad[15][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[1,2,1,ch,:]=np.sum(gradforw1_part3*(grad[16][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[1,2,2,ch,:]=np.sum(gradforw1_part3*(grad[17][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[2,0,0,ch,:]=np.sum(gradforw1_part3*(grad[18][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[2,0,1,ch,:]=np.sum(gradforw1_part3*(grad[19][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[2,0,2,ch,:]=np.sum(gradforw1_part3*(grad[20][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[2,1,0,ch,:]=np.sum(gradforw1_part3*(grad[21][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[2,1,1,ch,:]=np.sum(gradforw1_part3*(grad[22][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[2,1,2,ch,:]=np.sum(gradforw1_part3*(grad[23][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[2,2,0,ch,:]=np.sum(gradforw1_part3*(grad[24][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[2,2,1,ch,:]=np.sum(gradforw1_part3*(grad[25][:,:,:,ch,:].reshape(-1,f1)),axis=0)
			gradforw1[2,2,2,ch,:]=np.sum(gradforw1_part3*(grad[26][:,:,:,ch,:].reshape(-1,f1)),axis=0)

		return gradforw1

	def dE_db1(self):
		cost=self.cost
		label=self.label
		yhat=self.yhat
		w3=self.w3
		conv2=self.conv2
		w2=self.w2
		conv1=self.conv1
		b1=self.b1
		ch1=self.ch1
		f1=self.f1
		RFCar=self.RFCar
		dx1_db1=self.dx1_db1
		gradforb1_part1 = (self.dE_dout()*self.dout_dx3(self.w3.T)*self.dx3_drelu2()*self.drelu2_dx2()).reshape(RFCar[0],RFCar[1],RFCar[2],f1)
		gradforb1_part2 = self.gradforw1_part2cal(gradforb1_part1, self.dx2_drelu1(), f1)
		gradforb1_part3 = (gradforb1_part2 *self.drelu1_dx1())
		gradforb1=np.zeros((b1.shape))
		for i in range(0,8):
			dummy=np.zeros((conv1.shape))
			dummy[:,:,:,i]=dx1_db1[:,:,:,i]
			flat_db=np.expand_dims(dummy.ravel(),0)
			gradforb1[i]=sum((gradforb1_part3*flat_db).T)
		return gradforb1

	def backward(self):
		cost=self.cost
		label=self.label
		yhat=self.yhat
		w3=self.w3
		b3=self.b3
		w2=self.w2
		b2=self.b2
		w1=self.w1
		b1=self.b1
		conv2=self.conv2
		conv1=self.conv1
		dx1_dw1=self.dx1_dw1
		dx1_db1=self.dx1_db1
		dx2_dw2=self.dx2_dw2
		dx2_db2=self.dx2_db2
		ch1=self.ch1
		ch2=self.ch2
		f1=self.f1
		f2=self.f2
		RFCar=self.RFCar
		FC1=self.FC1

		grad_w3=self.dE_dw3()
		
		grad_b3=self.dE_db3()
		
		grad_w2=self.dE_dw2()
		grad_b2=self.dE_db2()
		
		grad_w1=self.dE_dw1()
		grad_b1=self.dE_db1()
		return grad_w1,grad_b1,grad_w2,grad_b2,grad_w3,grad_b3



