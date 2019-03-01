import numpy as np

class Nonzero_FVS(object):
	def __init__(self,FVS):
		super(Nonzero_FVS, self).__init__()
		self.FVS=FVS
	
	'''Finding the cells which are occupied'''
	def nonzerofeatures(self):
		FVS=self.FVS
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

if __name__ == '__main__':
	FVS=np.random.rand(25,25,25,8)
	a1=Nonzero_FVS(FVS)
	fv=a1.nonzerofeatures()
