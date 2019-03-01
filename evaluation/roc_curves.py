import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

class ROC_Curves(object):
	"""docstring for ROC_curves"""
	def __init__(self, emh_preds,emh_grndtrth,curr_epoch,dst_path):
		super(ROC_Curves, self).__init__()
		self.emh_preds = emh_preds
		self.emh_grndtrth = emh_grndtrth
		self.curr_epoch = curr_epoch
		self.dst_path = dst_path
		
	# Import some data to play with
	def roc_curves(self):
		n_classes=len(self.emh_grndtrth)
		lw=2
		Y_test=np.array(self.emh_grndtrth)
		y_score=np.array(self.emh_preds)

		ytest_flat=[k for s1 in self.emh_grndtrth for k in s1]
		yscore_flat=[k for s1 in self.emh_preds for k in s1]
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(n_classes):
		    fpr[i], tpr[i], _ = roc_curve(np.array(Y_test[i]),np.array(y_score[i]))
		    roc_auc[i] = auc(fpr[i], tpr[i])

		# Compute micro-average ROC curve and ROC area
		fpr["micro"], tpr["micro"], _ = roc_curve(np.array(ytest_flat),np.array(yscore_flat))
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		# Compute macro-average ROC curve and ROC area

		# First aggregate all false positive rates
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(n_classes):
		    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

		# Finally average it and compute AUC
		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

		# Plot all ROC curves
		plt.figure()
		fig = plt.figure()
		plt.plot(fpr["micro"], tpr["micro"],
		         label='micro-average ROC curve (area = {0:0.2f})'
		               ''.format(roc_auc["micro"]),
		         color='deeppink', linestyle=':', linewidth=4)

		plt.plot(fpr["macro"], tpr["macro"],
		         label='macro-average ROC curve (area = {0:0.2f})'
		               ''.format(roc_auc["macro"]),
		         color='navy', linestyle=':', linewidth=4)

		colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
		classnames=['easy','medium','hard']
		for i, color in zip(range(n_classes), colors):
		    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
		             label='ROC curve of class {0} (area = {1:0.2f})'
		             ''.format(classnames[i], roc_auc[i]))

		plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic for easy medium and hard classes')
		plt.legend(loc="lower right")
		
		fig.savefig(self.dst_path+'roccurve_epoch_'+str(self.curr_epoch)+'.png')
		# plt.show()
