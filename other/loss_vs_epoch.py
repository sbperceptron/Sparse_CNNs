import numpy as np 
import glob
from batch_generator import *
import multiprocessing as mp
from Sparse_CNN import *
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def nn_model(single_worker):
	
	FVS,label,w1,w2,w3,b1,b2,b3,RFCar,signal,ch1,ch2,f1,f2=single_worker

	# time.sleep(random.randint(1,3))
	# 1st convolution
	start1=time.time()
	conv1,gradient_w1,grad_b1=spconv3DLayerForward(FVS,w1,b1,ch1,f1,RFCar=RFCar) # refer to sparse_CNN.py file
	relu1=ReLU(conv1) # making the negative values zero

	# 2nd convolution
	relu1=nonzerofeatures(relu1)
	conv2,gradient_w2,grad_b2=spconv3DLayerForward(relu1,w2, b2,ch2,f2,RFCar=RFCar)
	relu2 = ReLU(conv2)
	
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
		w1_grad,b1_grad,w2_grad,b2_grad,w3_grad,b3_grad=Backward_propogation(cost,label,output,conv1,conv2,FC1,w1,b1,w2,b2,w3,b3,
			gradient_w1,grad_b1,gradient_w2,grad_b2,ch1,ch2,f1,f2,RFCar)
		end2=time.time()
		bckwd_time=end2-start2
		return w1_grad,w2_grad,w3_grad,b1_grad,b2_grad,b3_grad,cost,fwd_time,bckwd_time

def epoch_loss_function(batchsize,objects_pos_easy_pcd,objects_pos_medium_pcd,objects_pos_hard_pcd,objects_neg_easy_pcd,objects_neg_medium_pcd,
		objects_neg_hard_pcd,RFCar,ch1,ch2,f1,f2,epochweights_path):
	
	epoch_loss=[]
	for epoch in range(1,100):
		path=epochweights_path+'epoch_'+str(epoch+1)+'.weights.npz'
		print "Loss values for epoch", epoch
		# Read in the positive and negatives
		# positive and negative pcs
		test_pos_easy_path = objects_pos_easy_pcd
		test_pos_medium_path = objects_pos_medium_pcd
		test_pos_hard_path = objects_pos_hard_pcd

		test_neg_easy_path = objects_neg_easy_pcd
		test_neg_medium_path = objects_neg_medium_pcd
		test_neg_hard_path = objects_neg_hard_pcd
		
		test_pos_easy_path = glob.glob(test_pos_easy_path)
		test_pos_medium_path = glob.glob(test_pos_medium_path)
		test_pos_hard_path = glob.glob(test_pos_hard_path)
		
		test_neg_easy_path = glob.glob(test_neg_easy_path)
		test_neg_medium_path = glob.glob(test_neg_medium_path)
		test_neg_hard_path = glob.glob(test_neg_hard_path)

		# print len(test_pos_easy_path),len(test_pos_medium_path),len(test_pos_hard_path),\
		# len(test_neg_easy_path),len(test_neg_medium_path),len(test_neg_hard_path)
		
		test_pos_easy_path.sort()
		test_pos_medium_path.sort()
		test_pos_hard_path.sort()

		test_neg_easy_path.sort()
		test_neg_medium_path.sort()
		test_neg_hard_path.sort()

		test_pos_easy_path=test_pos_easy_path
		test_pos_medium_path=test_pos_medium_path
		test_pos_hard_path=test_pos_hard_path

		test_neg_easy_path=test_neg_easy_path
		test_neg_medium_path=test_neg_medium_path
		test_neg_hard_path=test_neg_hard_path

		# print len(test_pos_easy_path),len(test_pos_medium_path),len(test_pos_hard_path),\
		# len(test_neg_easy_path),len(test_neg_medium_path),len(test_neg_hard_path)

		pos_velodynes_easy=test_pos_easy_path
		pos_velodynes_medium=test_pos_medium_path
		pos_velodynes_hard=test_pos_hard_path

		neg_velodynes_easy=test_neg_easy_path
		neg_velodynes_medium=test_neg_medium_path
		neg_velodynes_hard=test_neg_hard_path

		pos_labels_easy=[1 for i in range(0,len(pos_velodynes_easy))]
		pos_labels_medium=[1 for i in range(0,len(pos_velodynes_medium))]
		pos_labels_hard=[1 for i in range(0,len(pos_velodynes_hard))]

		neg_labels_easy=[-1 for i in range(0,len(neg_velodynes_easy))]
		neg_labels_medium=[-1 for i in range(0,len(neg_velodynes_medium))]
		neg_labels_hard=[-1 for i in range(0,len(neg_velodynes_hard))]

		velodynes_easy=pos_velodynes_easy+neg_velodynes_easy
		velodynes_medium=pos_velodynes_medium+neg_velodynes_medium
		velodynes_hard=pos_velodynes_hard+neg_velodynes_hard

		labels_easy=pos_labels_easy+neg_labels_easy
		labels_medium=pos_labels_medium+neg_labels_medium
		labels_hard=pos_labels_hard+neg_labels_hard

		# loading the weights from the weigths saved for that epoch
		wghts=np.load(path)
	 	weights1=wghts['w1']
	 	weights2=wghts['w2']
	 	weights3=wghts['w3']
	 	b1=wghts['b1']
	 	b2=wghts['b2']
	 	b3=wghts['b3']
	 	emh=0
		class_losses=[]
		for velodynes,labels in zip([velodynes_easy,velodynes_medium,velodynes_hard],[labels_easy,labels_medium,labels_hard]):
			total_number_batches=len(velodynes)//batchsize
			curr_batch=1
			count=0
			loss_value=0
			loss_values_list=[]
			if emh==0:
				print "\t Processing the easy test cases"
				print "\t Total number of batches:", total_number_batches
			elif emh==1:
				print "\t Processing the medium test cases"
				print "\t Total number of batches:", total_number_batches
			else:
				print "\t Processing the hard test cases"
				print "\t Total number of batches:", total_number_batches
			emh+=1

			for fvs_batch,labels_batch,start_time in batch_feed_test(velodynes,labels,RFCar,batchsize): 
				# print "Processing Batch "+str(curr_batch)+" of "+str(total_number_batches)
				curr_batch+=1
				feature_extractor__end_time = time.time()
				work=[]
				for i,j in zip(fvs_batch,labels_batch):
					s=True
					work.append([i,j,weights1,weights2,weights3,b1,b2,b3,RFCar,s,ch1,ch2,f1,f2])
				
				pool=mp.Pool(processes=8)
				results1=pool.map(nn_model,work) # nn function in the begining of file
				pool.close()
				pool.join()
				
				results1=np.array(results1)
				total_loss=sum(results1[:,1])/(results1.shape[0])
				end_time=time.time()
				count+=1
				print count,
				loss_value+=total_loss
				# print "\t Time elapsed for processing a Batch:",np.round((end_time-start_time),2)
			print "\n"
			print loss_value/total_number_batches
			class_losses.append(loss_value/total_number_batches)
		epoch_loss.append(class_losses)
		
		print "\t The loss value for the easy test set:",np.round((epoch_loss[epoch-1][0]),2)
		print "\t The loss value for the medium test set:",np.round((epoch_loss[epoch-1][1]),2)
		print "\t The loss value for the hard test set:",np.round((epoch_loss[epoch-1][2]),2)
	return epoch_loss

	# plot the loss values
if __name__ == '__main__':
	epochweights_path="/home/saichand/3D_CNN/vote3deep2layer/train/pedestrian_weights/"
	
	objects_pos_easy_pcd="/home/saichand/3D_CNN/vote3deep2layer/data/crops/test/split/positive/Pedestrian/easy/*.bin"
	objects_pos_medium_pcd="/home/saichand/3D_CNN/vote3deep2layer/data/crops/test/split/positive/Pedestrian/medium/*.bin"
	objects_pos_hard_pcd="/home/saichand/3D_CNN/vote3deep2layer/data/crops/test/split/positive/Pedestrian/hard/*.bin"
	objects_neg_easy_pcd="/home/saichand/3D_CNN/vote3deep2layer/data/crops/test/split/negative/Pedestrian/easy/*.bin"
	objects_neg_medium_pcd="/home/saichand/3D_CNN/vote3deep2layer/data/crops/test/split/negative/Pedestrian/medium/*.bin"
	objects_neg_hard_pcd="/home/saichand/3D_CNN/vote3deep2layer/data/crops/test/split/negative/Pedestrian/hard/*.bin"
	batchsize=16
	RFCar=[int(2.0/0.2),int(2.0/0.2),int(3.0/0.2)]
	f1=8 # Refer to fig 3 in vote3deep
	ch1=6 # number of features for each grid cell
	f2=8 # Refer to fig 3 in vote3deep
	ch2=8 # number of features for each grid cell
	
	epoch_loss=epoch_loss_function(batchsize,objects_pos_easy_pcd,objects_pos_medium_pcd,objects_pos_hard_pcd,objects_neg_easy_pcd,objects_neg_medium_pcd,
		objects_neg_hard_pcd,RFCar,ch1,ch2,f1,f2,epochweights_path)

	loss_path="/home/saichand/3D_CNN/vote3deep2layer/train/performance/epoch_score_values.txt"
	if os.path.exists(loss_path):
		os.remove(loss_path)
	for epoch,loss in enumerate(epoch_loss):
		with open(loss_path, "a") as f: 
				f.write(str(epoch)+","+str(loss)+"\n") 
	epoch_loss=np.array(epoch_loss)
	t = np.linspace(0, epoch_loss.shape[0], num=epoch_loss.shape[0])

	

	
	plt.plot(t, epoch_loss[:,0], 'r', label="easy") # plotting t, a separately 
	plt.plot(t, epoch_loss[:,1], 'b', label="medium") # plotting t, b separately 
	plt.plot(t, epoch_loss[:,2], 'g', label="hard") # plotting t, c separately
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Loss value with every epoch')
	plt.legend(loc="upper right")
	plt.show()

