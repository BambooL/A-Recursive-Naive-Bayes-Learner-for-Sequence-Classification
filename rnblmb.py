import numpy as np
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
import Queue
import math
# class NBnode:
	
# 	clf = MultinomialNB()

# 	def __init__(self, id, data):
# 		self.id = id + 1
# 		self.CDML = getCDML(data)
# 		self.data = data
# 		NBnode self.left
# 		NBnode self.right

	
# 	def getCDML(data):
# 		return 1





# 	def generateNode(CDML, data):
# 		CDML_new = getCDML(data)
# 		if (CDML_new > CDML): 
# 			NBnode.clf.fit(train, label)
		
# 			prediction = NBnode.clf.predict(train)

# 			class1 = instance[prediction="false"]
# 			class2 = instance[prediction="true"]
		
# 			self.left = NBnode(this.id, class1)
# 			self.right = NBnode(this.id, class2)



# load data


def getCDML(dataset, label):

	global h_size
	C = 2
	t = 0
	f = 0
	for k in range(len(dataset)):
		if label[i] == 'false':
			f ++
		else:
			t ++

	for j in range(len(dataset)):
		if label[j] == 'false':
			P_cj = f/(f+t)
		else:
			P_cj = t/(f+t)

		each = P_cj * 

	CLL = len(dataset) * (LC1 + LC2)



	return CLL - math.log(len(dataset))/2 * h_size

	

def fit1node(train, label):
	global prev_CDML
	global clf_tree 
	global tree
	global h_size
	new_CDML = getCDML(train,label)

	if (new_CDML > prev_CDML):
		clf = MultinomialNB()
		train = np.array(train,dtype=float)
		clf.fit(train, label)
		clf_tree.append(clf)
		tree.append("1")
		h_size = h_size +1
		prediction = clf.predict(train)

		prev_CDML = new_CDML

		child1 = []
		child1_label = []
		child2 = []
		child2_label = []

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

# load data
csvReader = csv.reader(open('train/acq.csv', 'rb'), delimiter=',')
data = list(csvReader)
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

prev_CDML = 0
clf_tree = []
tree =[]
h_size = 3
global h_size
global prev_CDML
global clf_tree 
global tree 

# level order construct tree
# clf_tree = [MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), None, MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), None, None, None]
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




false = []
false_ground = []
true = []
true_ground = []
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

FF = 0
FT = 0
TF = 0
TT = 0

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


print FF
print FT
print TF
print TT
# print false_ground





		

















# cross-validation
# classifier



	

		


