import numpy as np 
# from Sparse_CNN import *
from Sparse_CNN import *

class NN1layer(object):
	"""docstring for NN1layer"""
	def __init__(self, single_worker):
		super(NN1layer, self).__init__()
		self.single_worker = single_worker
		
	'''A single forward and backward propogation operation. The Neural Network Architecture'''
	# total number of works =128
	# num of processors =8
	# batch size =16
	# each of the work is processed on one processor.

	def nn_model(self):
		
		FVS,label,w1,w2,b1,b2,RFCar,signal,ch1,f1=self.single_worker

		# time.sleep(random.randint(1,3))
		# 1st convolution
		start1=time.time()
		conv1,gradient_w1,grad_b1=spconv3DLayerForward(FVS,w1,b1,RFCar=RFCar) # refer to sparse_CNN.py file
		relu1=ReLU(conv1) # making the negative values zero

		# 2nd convolution- convolution operation using a filter of size equal to receptive field 
		# flattening the relu activation output
		FC1=np.expand_dims(relu1.ravel(),0)

		conv2=FC1.dot(w2)
		output=conv2+b2
		output=np.asscalar(output)
		# output=ReLU(conv2)
		# print output
		
		# ##################################################################################
		# # backward propogation
		# ###################################################################################
		cost = Linear_hinge_loss(output,label) # refer to Spasre CNN
		end1=time.time()
		fwd_time=end1-start1

		# print cost
		# print "\n"
		# # gradients for layer 2
		# weights 2 gradient
		if signal:
			return cost,output
		else:
			start2=time.time()
			w1_grad,b1_grad,w2_grad,b2_grad=Backward_propogation(cost,label,output,conv1,FC1,w1,b1,w2,b2,gradient_w1,grad_b1)
			end2=time.time()
			bckwd_time=end2-start2
			return w1_grad,w2_grad,b1_grad,b2_grad,cost,output,fwd_time,bckwd_time

