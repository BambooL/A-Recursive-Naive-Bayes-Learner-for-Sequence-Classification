#----------------------------------------------------------------------
#  RNBL-MB Implementation
#  Original Paper: Kang, D-K., Silvescu, A. and Honavar, V. (2006).
#  RNBL-MN: A Recursive Naive Bayes Learner for Sequence Classification. 
#  In: Proceedings of the Tenth Pacific-Asia Conference on Knowledge 
#  Discovery and Data Mining (PAKDD 2006). Springer-Verlag Lecture Notes 
#  in Computer Science Vol 3918, pp. 45-54.
#  Author: Xiao Liu
#  College of Information Sciences and Technologies
#  The Pennsylvania State University
#  last revised: 24 March 2015
#----------------------------------------------------------------------

import numpy as np
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
import Queue
import math
import sys

#----------------------------------------------------------------------
# get the CDML for the current node
def getCDML(train, label, clf):

	global h_size

	p_ij1 = clf.feature_log_prob_[0]
	p_ij2 = clf.feature_log_prob_[1]

	
	s = 0
	for j in range(len(train)):
		
		times1 = 1
		times2 = 1
		for i in range(len(p_ij1)):
			times1 = times1 * math.pow(math.exp(p_ij1[i]), train[j][i])
			times2 = times2 * math.pow(math.exp(p_ij2[i]), train[j][i])

		if (clf.predict(train[j]) == "false"):
			A = math.exp(clf.class_log_prior_[0]) * times1
		else:
			A = math.exp(clf.class_log_prior_[1]) * times2

		B = math.exp(clf.class_log_prior_[0]) * times1 + math.exp(clf.class_log_prior_[1]) * times2

		if (B != 0):
			if ((A/B)>0):
				s = s + math.log(A/B)

	CLL = len(train) * s

	return CLL - math.log(len(train))/2 * (h_size +2)


#----------------------------------------------------------------------
# Multinomial classifier fitted in one node
# Similar to any decision tree model, two most important values
# are the split criteria and the stop criteria. Here, the split
# criteria is that the CDML is larger than previous one and the
# stop criteria is that each node has at least 1 instance.
def fit1node(train, label):
	global prev_CDML
	global clf_tree 
	global tree
	global h_size
	child1 = []
	child1_label = []
	child2 = []
	child2_label = []
	
	clf = MultinomialNB()

# each node has at least 1 instance
	if (len(train) < 2):
		clf_tree.append(None)
		tree.append("#")
		return child1,child1_label,child2,child2_label

	train = np.array(train,dtype=float)

	clf.fit(train, label)
	
	
	new_CDML = getCDML(train,label,clf)

# Decide whether to split or not
	if (new_CDML > prev_CDML):
		clf_tree.append(clf)

		tree.append("1")
		h_size = h_size + 2

		prediction = clf.predict(train)

		prev_CDML = new_CDML



		for i in range(len(train)):
			if (prediction[i] == "false"):
				child1.append(train[i])
				child1_label.append(label[i])

			else:
				child2.append(train[i])
				child2_label.append(label[i])


	else:
		child1 = []
		child1_label = []
		child2 = []
		child2_label = []
		clf_tree.append(None)
		tree.append("#")

	return child1,child1_label,child2,child2_label


#----------------------------------------------------------------------
# Declaration
global h_size
global prev_CDML
global clf_tree 
global tree

# Load data
filename = sys.argv[1]
filename = "train/"+ filename + ".csv"
csvReader = csv.reader(open(filename, 'rb'), delimiter=',')
data = list(csvReader)

# Split data to train and test data sets
train = []
train_label = []
test = []
test_label = []
for i in range(1,int(len(data)*0.75)):
	train.append(data[i][1:])
	train_label.append(data[i][0])

for i in range(int(len(data)*0.75), len(data)):
	test.append(data[i][1:])
	test_label.append(data[i][0])


#----------------------------------------------------------------------
# Initial the values: 
# CDML is a very small number that root node can be generated; 
# List implement tree
# h is the value defined in the paper, initial as 1
prev_CDML = -99999999999
clf_tree = []
tree =[]
h_size = 1


#----------------------------------------------------------------------
# level order tree construction
# use a queue to do level-order traversal, once the CDML
# of the current node is larger than a previous one, fit
# the classifier in this node; otherwise, make it a None.
# output 
# tree = ['1', '1', '1', '1', '1', '#', '#', '#', '1', '#', '#', '#', '#']
q = Queue.Queue()
q.put(train)
q.put(train_label)
point = 0
length = len(clf_tree)
while (not q.empty()):
	size = q.qsize()/2

	for i in range(size):
		train = q.get()
		train_label = q.get()
		c1,l1,c2,l2 = fit1node(train, train_label)

		if (tree[-1]!= "#"):
			q.put(c1)
			q.put(l1)
			q.put(c2)
			q.put(l2)


print tree


# begin test
# initial the prediction set
false = []
false_ground = []
true = []
true_ground = []


#----------------------------------------------------------------------
# level-order data splition
# At begining, the whole dataset is in the root node. The fitted
# classifier will predict each instance a class and the dataset
# is splited to two. 
q = Queue.Queue()
q.put(test)
q.put(test_label)
k = 0

while (not q.empty() and k < len(clf_tree)):
	size = q.qsize()/2

	for i in range(size):
		test = q.get()
		test_label = q.get()
		if (clf_tree[k] == None):
			if (k % 2 == 0):
				true.append(test)
				true_ground.append(test_label)
			else:
				false.append(test)
				false_ground.append(test_label)
			k = k+1

		else:
			child1 = []
			child1_label = []
			child2 = []
			child2_label = []
			clf = clf_tree[k]
			test = np.array(test,dtype=float)
			test_prediction = clf.predict(test)
			for i in range(len(test)):
				if (test_prediction[i] == "false"):
					child1.append(test[i])
					child1_label.append(test_label[i])

				else:
					child2.append(test[i])
					child2_label.append(test_label[i])
			q.put(child1)
			q.put(child1_label)
			q.put(child2)
			q.put(child2_label)
			k = k+1


#----------------------------------------------------------------------
# Result Analysis
# Confusion matrix construction

FF = 0.0
FT = 0.0
TF = 0.0
TT = 0.0


for i in range(len(false_ground[0])):
	if (false_ground[0][i] != "false"):
		FT = FT + 1
	else:
		FF = FF + 1

for i in range(len(true_ground[0])):
	if (true_ground[0][i] != "true"):
		TF = TF + 1
	else:
		TT = TT + 1

# Measurements
print "Accuracy =" 
print (TT + FF) /(TT + FF + TF + FT)
print "Precison ="
print FF / (FF + FT)
print "Recall ="
print FF / (FF + TF)
print "F-measure ="
print 2* (FF/(FF + FT)) *(FF / (FF + TF)) / (FF / (FF + TF) + FF / (FF + FT))

#----------------------------------------------------------------------



	

		


