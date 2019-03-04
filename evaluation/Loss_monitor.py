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

class Loss_Monitor(object):
	def __init__(self,loss_path):
		self.loss_path=loss_path

	def animate1(self,k):
		# loss acc time
		pullData = open(self.loss_path,"r").read()
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
		newax1=ax1.twinx()
		newax1.set_ylabel('Accuracy', color='y')
		newax1.clear()
		newax1.plot(yar,xar2,color='y',label='Accuracy')
		

	def animate2(self,k):
		pullData = open(self.loss_path,"r").read()
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
		
	def loss_monitor(self):
		ani1 = animation.FuncAnimation(fig1, animate1, interval=1)
		ani2 = animation.FuncAnimation(fig2, animate2, interval=1)
		plt.show()


ap = argparse.ArgumentParser()
ap.add_argument("-l", "--losspath", required=True,
	help="type [int] \n \
	[INFO] The path to the lossvalues")
se=Loss_Monitor(agrs["losspath"])
se.loss_monitor()
args = vars(ap.parse_args())