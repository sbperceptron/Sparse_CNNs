import numpy as np 
import os,glob,time

class Sparse_CNN:
	def __init__(self,inchannels,outchannels,filtersize,
		dimensions,bias=True,mode="fan_in",nonlinearity="relu"):
		self.inchannels=inchannels
		self.outchannels=outchannels
		self.filtersize=filtersize
		self.dimensions=dimensions
		self.bias=bias
		self.wghts_init_method=wghts_init_method
		self.mode=mode
		self.nonlinearity=nonlinearity
	
	def Kaiming_normal(self):
		weights=nn.init.kaiming_normal_(wght,
			self.mode=mode,self.nonlinearity=nonlinearity)
		return weights

	def forward(self,input,weights,init=True):
		# input is a feature vector of shape equal to receptive 
		# field size of the object
		if (init==True):
			weights=Kaiming_normal()


	def backward(self,):




if __name__ == '__main__':
	wght=torch.empty(self.outchannels,
		self.inchannels,self.filtersize,self.filter_size,
		self.filter_size)# fig 3 from paper w1