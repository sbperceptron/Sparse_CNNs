# import data as dt
from help_functions import *
from feature_extractor import *
from sklearn.metrics import hinge_loss
from scipy.signal import convolve
# import pycuda.autoinit
# import pycuda.driver as cuda
# import pycuda.gpuarray as gpuarray
import time
# from pycuda.compiler import SourceModule
# import skcuda.linalg as linalg
# from numba import guvectorize


'''Finding the cells which are occupied'''
def nonzerofeatures(FVS):
	final_FVS=dict()
	# print FVS.shape
	allbool=np.logical_or(FVS[:,:,:,0]!=0,np.logical_or(FVS[:,:,:,1]!=0,FVS[:,:,:,2]!=0))
	allbool=np.logical_or(allbool, np.logical_or(FVS[:,:,:,3]!=0,FVS[:,:,:,4]!=0))
	allbool=np.logical_or(allbool,np.logical_or(FVS[:,:,:,5]!=0,FVS[:,:,:,6]!=0))
	allbool=np.logical_or(allbool,FVS[:,:,:,7]!=0)
	loc_fv=np.where(allbool)
	loc=np.stack((loc_fv[0],loc_fv[1],loc_fv[2]),axis=1)
	# print loc.shape
	for i in loc:
		final_FVS[(i[0], i[1], i[2])]=FVS[i[0],i[1],i[2]]

	return final_FVS

'''Caliculating the RelU activation Section 3 B'''
def ReLU(x):
	mask  = (x >0) * 1.0 
	return mask * x

'''Caliculating the differentiation of RelU values for input'''
def d_ReLU(x):
	mask  = (x >0) * 1.0 
	return mask 

'''Given the output score and label caliculate the L1 hinge loss Section 4 A'''
def Linear_hinge_loss(yhat,y):
	if (yhat>1 and y==1) or (yhat <-1 and y==-1):
		loss=0
	else :
		loss=max(0.0,1-yhat*y)
	
	return loss

'''The Sparse Convolution or the voting operation Section 3 A'''
def spconv3DLayerForward(FVS,weights,bias,ch,f,RFCar=None):
	
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
#			FV=FVS[cell]
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


def dE_dout(cost,label,yhat):
	if (label*yhat) < 1:
		return -label
	else:
		return 0

def dout_dx3(w3):
	return w3

def dx3_drelu2():
	return 1

def drelu2_dx2(conv2):
	flat_conv2=np.expand_dims(conv2.ravel(),0)
	return d_ReLU(flat_conv2)

def dx2_drelu1(w2):
	return w2
	
def drelu1_dx1(conv1):
	flat_conv1=np.expand_dims(conv1.ravel(),0)
	return d_ReLU(flat_conv1)

def dout_dw3(fc):
	return fc

def dE_dw3(cost,label,yhat,fc):
	gradforw3=dE_dout(cost,label,yhat)*dout_dw3(fc)
	return gradforw3

def dE_db3(cost,label,yhat):
	gradforb3=dE_dout(cost,label,yhat)
	return gradforb3

def dE_dw2(cost,label,yhat,w3,conv2,w2,grad,ch2,f2):
	gradforw2_part1=(dE_dout(cost,label,yhat)*dout_dx3(w3.T)*dx3_drelu2()*drelu2_dx2(conv2)).reshape(-1,f2)
	gradforw2=np.zeros((w2.shape))
	
	for ch in range(0,ch2):
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
		
	return gradforw2

def dE_db2(cost,label,yhat,w3,conv2,b2,dx2_db2,ch2,f2):
	gradforb2_part1=dE_dout(cost,label,yhat)*dout_dx3(w3.T)*dx3_drelu2()*drelu2_dx2(conv2)
	
	gradforb2=np.zeros((b2.shape))
	for i in range(0,8):
		dummy=np.zeros((conv2.shape))
		dummy[:,:,:,i]=dx2_db2[:,:,:,i]
		flat_db=np.expand_dims(dummy.ravel(),0)
		gradforb2[i]=sum((gradforb2_part1*flat_db).T)
	return gradforb2

def gradforw1_part2cal(gradforw1_part1, weights, RFCar,f):
	
	FVS=nonzerofeatures(gradforw1_part1)
	numfilters=weights.shape[-1]
	scores=np.zeros((RFCar[0],RFCar[1],RFCar[2],f))
	
	for i in range(0,weights.shape[-1]):
		score=np.zeros((RFCar[0],RFCar[1],RFCar[2]))
		weight=weights[:,:,:,:,i]
		weight=np.flip(weight,axis=0)
		weight=np.flip(weight,axis=1)
		weight=np.flip(weight,axis=2)

		for cell,FV in FVS.items():
			count=0
#			FV=FVS[cell]
			c=np.array(cell, dtype=int)
			
			if (c[0]+2) < (RFCar[0]) and (c[1]+2)<RFCar[1] and (c[2]+2)<RFCar[2] and c[0]!=0 and c[1]!=0 and c[2]!=0:
				
				score[c[0]-1:c[0]+2,c[1]-1:c[1]+2,c[2]-1:c[2]+2]+=np.sum(weight*FV, axis=-1)			
		
		scores[:,:,:,i]=score

	scores=np.expand_dims(scores.ravel(),0)
	
	return scores


def dE_dw1(cost,label,yhat,w3,conv2,w2,conv1,w1,grad,ch1,f1,RFCar):
	gradforw1_part1 = (dE_dout(cost,label,yhat)*dout_dx3(w3.T)*dx3_drelu2()*drelu2_dx2(conv2)).reshape(RFCar[0],RFCar[1],RFCar[2],f1)
	gradforw1_part2 = gradforw1_part2cal(gradforw1_part1, dx2_drelu1(w2), RFCar, f1)
	gradforw1_part3 = (gradforw1_part2 *drelu1_dx1(conv1)).reshape(-1,f1)
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

def dE_db1(cost,label,yhat,w3,conv2,w2,conv1,b1,dx1_db1,ch1,f1,RFCar):
	gradforb1_part1 = (dE_dout(cost,label,yhat)*dout_dx3(w3.T)*dx3_drelu2()*drelu2_dx2(conv2)).reshape(RFCar[0],RFCar[1],RFCar[2],f1)
	gradforb1_part2 = gradforw1_part2cal(gradforb1_part1, dx2_drelu1(w2), RFCar, f1)
	gradforb1_part3 = (gradforb1_part2 *drelu1_dx1(conv1))
	gradforb1=np.zeros((b1.shape))
	for i in range(0,8):
		dummy=np.zeros((conv1.shape))
		dummy[:,:,:,i]=dx1_db1[:,:,:,i]
		flat_db=np.expand_dims(dummy.ravel(),0)
		gradforb1[i]=sum((gradforb1_part3*flat_db).T)
	return gradforb1

def Backward_propogation(cost,label,yhat,conv1,conv2,FC1,w1,b1,w2,b2,w3,b3,dx1_dw1,dx1_db1,dx2_dw2,dx2_db2,ch1,ch2,f1,f2,RFCar):

	grad_w3=dE_dw3(cost,label,yhat,FC1)
	grad_b3=dE_db3(cost,label,yhat)
	
	grad_w2=dE_dw2(cost,label,yhat,w3,conv2,w2,dx2_dw2,ch2,f2)
	grad_b2=dE_db2(cost,label,yhat,w3,conv2,b2,dx2_db2,ch2,f2)
	
	grad_w1=dE_dw1(cost,label,yhat,w3,conv2,w2,conv1,w1,dx1_dw1,ch1,f1,RFCar)
	grad_b1=dE_db1(cost,label,yhat,w3,conv2,w2,conv1,b1,dx1_db1,ch1,f1,RFCar)
	
	return grad_w1,grad_b1,grad_w2,grad_b2,grad_w3,grad_b3



