from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
from itertools import cycle
import numpy as np
from sklearn.metrics import average_precision_score

class PR_Curves(object):
	"""docstring for PR_Curves"""
	def __init__(self, emh_preds,emh_grndtrth,curr_epoch,dst_path):
		super(PR_Curves, self).__init__()
		self.emh_preds = emh_preds
		self.emh_grndtrth = emh_grndtrth
		self.curr_epoch = curr_epoch
		self.dst_path = dst_path
		
	def pr_curves(self):
		colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
		
		Y_test=np.array(self.emh_grndtrth)
		y_score=np.array(self.emh_preds)
		ytest_flat=[k for s1 in self.emh_grndtrth for k in s1]
		yscore_flat=[k for s1 in self.emh_preds for k in s1]
		average_precision=[]
		precision=[]
		recall=[]
		n_classes=len(self.emh_grndtrth)

		# For each class
		precision = dict()
		recall = dict()
		average_precision = dict()

		for i in range(n_classes):
			precision[i], recall[i], _ = precision_recall_curve(np.array(Y_test[i]),
																np.array(y_score[i]))
			average_precision[i] = average_precision_score(np.array(Y_test[i]), np.array(y_score[i]))
		
		precision["micro"], recall["micro"], _ = precision_recall_curve(np.array(ytest_flat),np.array(yscore_flat))
		average_precision["micro"] = average_precision_score(np.array(ytest_flat),np.array(yscore_flat),
														 average="micro")
		print('Average precision score, micro-averaged over all classes: {0:0.2f}'
		  .format(average_precision["micro"]))
		
		plt.figure(figsize=(7, 8))
		f_scores = np.linspace(0.2, 0.8, num=4)
		lines = []
		labels = []
		
		fig = plt.figure()
		l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
		lines.append(l)
		labels.append('micro-average Precision-recall (area = {0:0.2f})'
					  ''.format(average_precision["micro"]))
		classnames=['easy','medium','hard']
		for i, color in zip(range(n_classes), colors):
			l, = plt.plot(recall[i], precision[i], color=color, lw=2)
			lines.append(l)
			labels.append('Precision-recall for {0} (area = {1:0.2f})'
						  ''.format(classnames[i], average_precision[i]))

		fig.subplots_adjust(bottom=0.25)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Precision-Recall curve for easy medium and hard')
		plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
		fig.savefig(self.dst_path+'prcurve_epoch_'+str(self.curr_epoch)+'.png')









