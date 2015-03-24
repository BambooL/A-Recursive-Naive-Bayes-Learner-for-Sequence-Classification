filename = ["acq", "corn", "earn", "interest", "ship", "crude", "trade", "grain", "money-fx", "wheat"]

for name in filename:
	train = name + ".train"
	arff = name + ".arff"
	ff1 = open(train, "r")
	while (1):
		line1 = ff1.readline()[0:-2]
		line2 = ff1.readline()[0:-2]
		line = line1 + "," + '\"'+ line2 + '\"\r\n'
		if (line1 == ''): 
			break
		ff2 = open(arff, "a+")
		ff2.write(line)        
		ff2.close()

filename = ["acq", "corn", "earn", "interest", "ship", "crude", "trade", "grain", "money-fx", "wheat"]

for name in filename:
	train = name + ".test"
	arff = name + ".arff"
	ff1 = open(train, "r")
	while (1):
		line1 = ff1.readline()[0:-2]
		line2 = ff1.readline()[0:-2]
		line = line1 + "," + '\"'+ line2 + '\"\r\n'
		if (line1 == ''): 
			break
		ff2 = open(arff, "a+")
		ff2.write(line)        
		ff2.close()

filename = ["acq", "corn", "earn", "interest", "ship", "crude", "trade", "grain", "money-fx", "wheat"]
for name in filename:
	trai = name + ".arff"
	train = open(trai, "r")
	tes = "../test/"+ name + ".arff"
	test = open(tes, "r")
	at = name + "_att.arff"
	att = open(at, "a+")
	l1 = "@relation " + name + "\r\n"
	l2 = "@attribute class {false, true} \r\n" + "@attribute text string \r\n" + "@data\r\n"
	att.write(l1)
	att.write(l2)
	for line in train.readlines():
		att.write(line)
	for line in test.readlines():
		att.write(line)







