import numpy as np 
import random

class Read_label_data(object):
	"""docstring for Read_label_data"""
	def __init__(self, label_path, objects=None,calib_path=None,is_velo_cam=None,proj_velo=None):
		super(Read_label_data, self).__init__()
		self.label_path = label_path
		self.calib_path=calib_path
		self.is_velo_cam=is_velo_cam
		self.proj_velo=proj_velo
		self.objects=objects
		
	def read_label_from_txt(self):
		label_path=self.label_path
		"""Read label from txt file."""
		text = np.fromfile(label_path)
		bounding_box = []
		with open(label_path, "r") as f:
			labels = f.read().split("\n")
			cdontcare=0
			for label in labels:
				if not label:
					continue
				label = label.split(" ")
				

				if label[0] == (self.objects): 
					bounding_box.append(label[8:15])

				if (label[0] == "DontCare") and cdontcare==0:
					bounding_box.append(label[8:15])
					cdontcare=1
					continue

				# if (label[0] == "DontCare"):
				# 	continue

				# if label[0] != ("Car"): 
				# 	bounding_box.append(label[8:15])


				# if label[0] == ("Pedestrian"): 
				# 	bounding_box.append(label[8:15])


				# if label[0] == ("Cyclist"): 
				# 	bounding_box.append(label[8:15])

				# if label[0] == ("Truck"): 
				# 	bounding_box.append(label[8:15])

		if bounding_box:

			data = np.array(bounding_box, dtype=np.float32)
			return data[:, 3:6], data[:, :3], data[:, 6]
		else:
			return None, None, None

	"""Read labels from xml or txt file.
		Original Label value is shifted about 0.27m from object center.
		So need to revise the position of objects.
		"""
	def read_label_data(self):
		places, size, rotates = self.read_label_from_txt()
		if places is None:
			return None, None, None
		rotates = np.pi / 2 - rotates
		dummy = np.zeros_like(places)
		dummy = places.copy()
		if self.calib_path:
			places = np.dot(dummy, self.proj_velo.transpose())[:, :3]
		else:
			places = dummy
		if self.is_velo_cam:
			places[:, 0] += 0.27

		return places, rotates, size