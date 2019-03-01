# import data as dt
from sklearn.metrics import hinge_loss
from scipy.signal import convolve
import time
import numpy as np
from .non_zerofvs import Nonzero_FVS


class Sparse_CNN_2layer_forward(object):
	"""docstring for Sparse_CNN_2layer"""
	def __init__(self, FVS, weights,bias,ch,f,RFCar=None):
		super(Sparse_CNN_2layer_forward, self).__init__()
		self.FVS = FVS
		self.weights=weights
		self.bias=bias
		self.ch=ch
		self.f=f
		self.RFCar=RFCar

	'''The Sparse Convolution or the voting operation Section 3 A'''
	def forward(self):
		FVS=self.FVS
		# print "\t",len(FVS.keys())
		weights=self.weights
		bias=self.bias
		ch=self.ch
		f=self.f
		RFCar=self.RFCar
		numfilters=weights.shape[-1]
		scores=np.zeros((RFCar[0],RFCar[1],RFCar[2],f))
		grad_w000=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w001=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w002=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w010=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w011=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w012=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w020=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w021=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w022=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w100=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w101=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w102=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w110=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w111=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w112=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w120=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w121=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w122=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w200=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w201=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w202=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w210=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w211=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w212=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w220=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w221=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))
		grad_w222=np.zeros((RFCar[0],RFCar[1],RFCar[2],ch,numfilters))

		grad_b=np.zeros((RFCar[0],RFCar[1],RFCar[2],numfilters))
		
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
				if (c[0]+2) < (RFCar[0]) and (c[1]+2)<RFCar[1] and (c[2]+2)<RFCar[2] and c[0]!=0 and c[1]!=0 and c[2]!=0:
					
					score[c[0]-1:c[0]+2,c[1]-1:c[1]+2,c[2]-1:c[2]+2]+=np.sum(weight*FV, axis=-1)			
					

					grad_w222[c[0]-1,c[1]-1,c[2]-1,:,i]=FV
					grad_w221[c[0]-1,c[1]-1,c[2],:,i]=FV
					grad_w220[c[0]-1,c[1]-1,c[2]+1,:,i]=FV
					grad_w212[c[0]-1,c[1],c[2]-1,:,i]=FV
					grad_w211[c[0]-1,c[1],c[2],:,i]=FV
					grad_w210[c[0]-1,c[1],c[2]+1,:,i]=FV
					grad_w202[c[0]-1,c[1]+1,c[2]-1,:,i]=FV
					grad_w201[c[0]-1,c[1]+1,c[2],:,i]=FV
					grad_w200[c[0]-1,c[1]+1,c[2]+1,:,i]=FV
					grad_w122[c[0],c[1]-1,c[2]-1,:,i]=FV
					grad_w121[c[0],c[1]-1,c[2],:,i]=FV
					grad_w120[c[0],c[1]-1,c[2]+1,:,i]=FV
					grad_w112[c[0],c[1],c[2]-1,:,i]=FV
					grad_w111[c[0],c[1],c[2],:,i]=FV
					grad_w110[c[0],c[1],c[2]+1,:,i]=FV
					grad_w102[c[0],c[1]+1,c[2]-1,:,i]=FV
					grad_w101[c[0],c[1]+1,c[2],:,i]=FV
					grad_w100[c[0],c[1]+1,c[2]+1,:,i]=FV
					grad_w022[c[0]+1,c[1]-1,c[2]-1,:,i]=FV
					grad_w021[c[0]+1,c[1]-1,c[2],:,i]=FV
					grad_w020[c[0]+1,c[1]-1,c[2]+1,:,i]=FV
					grad_w012[c[0]+1,c[1],c[2]-1,:,i]=FV
					grad_w011[c[0]+1,c[1],c[2],:,i]=FV
					grad_w010[c[0]+1,c[1],c[2]+1,:,i]=FV
					grad_w002[c[0]+1,c[1]+1,c[2]-1,:,i]=FV
					grad_w001[c[0]+1,c[1]+1,c[2],:,i]=FV
					grad_w000[c[0]+1,c[1]+1,c[2]+1,:,i]=FV

			grad_b[:,:,:,i]=np.where(score!=0, 1, score)
			score=np.where(score!=0, score+bias[i], score)

			scores[:,:,:,i]=score
			
		gradient_w1 = [grad_w000,grad_w001,grad_w002,grad_w010,grad_w011\
		,grad_w012,grad_w020,grad_w021,grad_w022,grad_w100,grad_w101,\
		grad_w102,grad_w110,grad_w111,grad_w112,grad_w120,grad_w121,\
		grad_w122,grad_w200,grad_w201,grad_w202,grad_w210,grad_w211,\
		grad_w212,grad_w220,grad_w221,grad_w222]

		return scores,gradient_w1,grad_b 




