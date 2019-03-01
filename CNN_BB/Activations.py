import numpy as np 

class Activation_Functions:
	"""docstring for Activation_Functions"""
	def __init__(self, x):
		self.x = x
		
	'''Caliculating the RelU activation Section 3 B'''
	def relu(self):
		x=self.x
		mask  = (x >0) * 1.0 
		return mask * x

	'''Caliculating the differentiation of RelU values for input'''
	def d_relu(self):
		x=self.x
		mask  = (x >0) * 1.0 
		return mask 

if __name__ == '__main__':
	x=np.random.rand(4,4)
	ob1=Activation_Functions(x)
	r=ob1.relu()