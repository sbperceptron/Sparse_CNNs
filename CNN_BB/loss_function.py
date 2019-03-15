import numpy as np 

class Loss_Functions:
	"""docstring for Loss_Functions"""
	def __init__(self, yhat, ytrue):
		self.yhat = yhat
		self.ytrue = ytrue

	'''Given the output score and label caliculate the L1 hinge loss Section 4 A'''
	def Linear_hinge_loss(self):
		loss=max(0.0,1-self.yhat*self.ytrue)
		return loss


