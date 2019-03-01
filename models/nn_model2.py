import numpy as np 
import time
from CNN_BB import Sparse_CNN_2layer_backward
from CNN_BB import Sparse_CNN_2layer_forward
from CNN_BB import Activation_Functions,Loss_Functions
from CNN_BB import Nonzero_FVS

def nn_model2(single_worker):
	
	FVS,label,w1,w2,w3,b1,b2,b3,RFCar,signal,ch1,ch2,f1,f2=single_worker

	# time.sleep(random.randint(1,3))
	# 1st convolution
	start1=time.time()
	conv1_obj=Sparse_CNN_2layer_forward(FVS,w1,b1,ch1,f1,RFCar=RFCar)
	conv1,gradient_w1,grad_b1=conv1_obj.forward() # refer to sparse_CNN.py file
	relu1_obj=Activation_Functions(conv1)
	relu1=relu1_obj.relu() # making the negative values zero

	# 2nd convolution
	nz_obj=Nonzero_FVS(relu1)
	relu1=nz_obj.nonzerofeatures()
	conv2_obj=Sparse_CNN_2layer_forward(relu1,w2,b2,ch2,f2,RFCar=RFCar)
	conv2,gradient_w2,grad_b2=conv2_obj.forward()

	relu2_obj=Activation_Functions(conv2)
	relu2 = relu2_obj.relu()

	# 3rd convolution- convolution operation using a filter of size equal to receptive field 
	# flattening the relu activation output
	FC1=np.expand_dims(relu2.ravel(),0)

	
	conv3=FC1.dot(w3)
	output=conv3+b3
	output=np.asscalar(output)
	# output=ReLU(conv2)
	# print output

	# ##################################################################################
	# # backward propogation
	# ###################################################################################
	l1hloss=Loss_Functions(output,label)
	cost = l1hloss.Linear_hinge_loss() # refer to Spasre CNN
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
		bckwd_obj=Sparse_CNN_2layer_backward(cost,label,output,conv1,conv2,FC1,w1,b1,w2,b2,w3,b3,
			gradient_w1,grad_b1,gradient_w2,grad_b2,ch1,ch2,f1,f2,RFCar)
		w1_grad,b1_grad,w2_grad,b2_grad,w3_grad,b3_grad=bckwd_obj.backward()
		end2=time.time()
		bckwd_time=end2-start2
		return w1_grad,w2_grad,w3_grad,b1_grad,b2_grad,b3_grad,cost,output,fwd_time,bckwd_time
	