import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from math import log10
import numpy as np

fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)
No_of_batches=13016
time_elapsed_for_a_batch=2

def animate1(k):
	loss_path="/home/pyimagesearch/3D_Sparse_CNN/train_singlelayer/performance/loss_values_train.txt"
	# loss acc time
	pullData = open(loss_path,"r").read()
	dataArray = pullData.split('\n')
	xar1 = []
	xar2 = []
	yar = []
	error_sum1=0
	error_sum2=0
	plot_lines = []

	for i,eachLine in enumerate(dataArray):
		if len(eachLine)>1:
			x,y,t = eachLine.split(',')
			error_sum1+=float(x)
			yar.append(float(t))
			xar1.append(log10(float(error_sum1/(i+1))))
			error_sum2+=float(y)
			xar2.append(float(error_sum2/(i+1)))
	ax1.clear()
	ax1.plot(yar,xar1,color='k',label='Loss')
	ax1.set_title("Training: Loss(moving average of Log(loss) and accuracy vs Time) ")
	ax1.set_xlabel('Time (seconds)')
	ax1.set_ylabel('Error (log(error))', color='k')
   	# ax1.tick_params(axis='y', labelcolor='k')
   	
	newax1=ax1.twinx()
	newax1.set_ylabel('Accuracy', color='y')
	newax1.clear()
	newax1.plot(yar,xar2,color='y',label='Accuracy')
	# newax1.tick_params(axis='y', labelcolor='y')
	
	# epochs=len(dataArray)//No_of_batches
	# xposition = range(0,int(No_of_batches*(epochs+1)*time_elapsed_for_a_batch),int(No_of_batches*time_elapsed_for_a_batch))
	# colors=['b','g','r','c','m','y','k']
	# for j,xc in enumerate(xposition):
	#	 ax1.axvline(x=xc, color=colors[np.random.randint(0,6)],label='epoch_ '+str(j),linestyle='--',linewidth=2.0)
	#	 ax1.legend(loc=0)

def animate2(k):
	loss_path="/home/pyimagesearch/3D_Sparse_CNN/train_singlelayer/performance/loss_values_test.txt"
	pullData = open(loss_path,"r").read()
	dataArray = pullData.split('\n')
	xar1 = []
	xar2=[]
	yar = []
	error_sum1=0
	error_sum2=0
	for i,eachLine in enumerate(dataArray):
		if len(eachLine)>1:
			x,y,t = eachLine.split(',')
			error_sum1+=float(x)
			yar.append(float(t))
			xar1.append(log10(float(error_sum1/(i+1))))
			error_sum2+=float(y)
			xar2.append(loat(error_sum2/(i+1)))
	ax2.clear()
	ax2.plot(yar,xar1,color='k',label='Loss')
	ax2.set_title("Testing: Loss(moving average of Log(loss) and accuracy vs Time) ")
	ax2.set_xlabel('Time (seconds)')
	ax2.set_ylabel('Error (log(error))',color='k')
	# ax2.tick_params(axis='y', labelcolor='k')
	newax2.clear()
	newax2=ax2.twinx()
	newax2.set_ylabel('Accuracy',color='y')
	newax2.plot(yar,xar2,color='y',label='Accuracy')
	# newax2.tick_params(axis='y', labelcolor='y')

	# epochs=len(dataArray)//No_of_batches
	# xposition = range(0,int(No_of_batches*(epochs+1)*time_elapsed_for_a_batch),int(No_of_batches*time_elapsed_for_a_batch))
	# colors=['b','g','r','c','m','y','k']
	# for j,xc in enumerate(xposition):
	# 	ax2.axvline(x=xc, color=colors[np.random.randint(0,6)],label='epoch_ '+str(j), linestyle='--',linewidth=2.0)
	# 	ax2.legend(loc=0)


ani1 = animation.FuncAnimation(fig1, animate1, interval=1)

ani2 = animation.FuncAnimation(fig2, animate2, interval=1)
# plt.tight_layout()
plt.show()



# x,y,t = eachLine.split(',')
# yar.append(float(t))
# if float(x)<=0.001:
# 	xar1.append(log10(float(0.01)))
# else:
# 	xar1.append(log10(float(x)))

# xar2.append(float(y))