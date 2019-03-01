import numpy as np 
import random

class Read_Calib_File:
	def __init__(self,calib_path):
		self.calib_path=calib_path

	def read_calib_file(self):
		"""Read a calibration file."""
		data = {}
		calib_path=self.calib_path
		with open(calib_path, 'r') as f:
			for line in f.readlines():
				if not line or line == "\n":
					continue
				key, value = line.split(':', 1)
				try:
					data[key] = np.array([float(x) for x in value.split()])
				except ValueError:
					pass
		return data

class Proj_to_velo(object):
	"""docstring for Proj_to_velo"""
	def __init__(self, calib_data):
		super(Proj_to_velo, self).__init__()
		self.calib_data = calib_data
		
	def proj_to_velo(self):
		calib_data=self.calib_data
		"""Projection matrix to 3D axis for 3D Label"""
		rect = calib_data["R0_rect"].reshape(3, 3)
		velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
		inv_rect = np.linalg.inv(rect)
		inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
		return np.dot(inv_velo_to_cam, inv_rect)
