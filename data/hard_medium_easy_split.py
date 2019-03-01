import numpy as np 
import os
from shutil import copyfile
from help_functions import *

def create_easy_medium_hard_split_pos(pths4,root,dirs_easy_medium_hard_crops):
	split=[[0,1,2],[3,4,5],[6,7,8]]
	
	for c,objectpcpaths in enumerate(pths4):
		c1=0
		c2=0
		c3=0
		for k,pcpath in enumerate(objectpcpaths):
			# if k%100 == 0 and k!=0:
			# 	print "processing file " + str(k) +" of "+str(l) 
			pc=load_binfile(pcpath)
			if pc.shape[0]<50:
				if not os.path.exists(root+dirs_easy_medium_hard_crops[split[c][2]]):
					os.makedirs(root+dirs_easy_medium_hard_crops[split[c][2]])
				
				name='/'+ '%6d'%c1+'.bin'
				copyfile(pcpath,root+dirs_easy_medium_hard_crops[split[c][2]]+name)
				c1+=1
			if pc.shape[0]>50 and pc.shape[0]<150:
				if not os.path.exists(root+dirs_easy_medium_hard_crops[split[c][1]]):
					os.makedirs(root+dirs_easy_medium_hard_crops[split[c][1]])
				
				name='/'+ '%6d'%c2+'.bin'
				copyfile(pcpath,root+dirs_easy_medium_hard_crops[split[c][1]]+name)
				c2+=1
			if pc.shape[0]>150:
				if not os.path.exists(root+dirs_easy_medium_hard_crops[split[c][0]]):
					os.makedirs(root+dirs_easy_medium_hard_crops[split[c][0]])
				
				name='/'+ '%6d'%c3+'.bin'
				copyfile(pcpath,root+dirs_easy_medium_hard_crops[split[c][0]]+name)
				c3+=1
		


