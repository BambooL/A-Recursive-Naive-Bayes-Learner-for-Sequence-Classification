filename = ["acq", "corn", "earn", "interest", "ship", "crude", "trade", "grain", "money-fx", "wheat"]

for name in filename:
	train = name + ".train"
	arff = name + ".arff"
	ff1 = open(train, "r")
	ff2 = open(arff, "a+")
	ff2.write("@relation " + name+ '\"\r\n')
	ff2.write("@attribute class {true, false}"+ '\"\r\n')
	ff2.write("@attribute text String"+ '\"\r\n')
	ff2.write("@data"+ '\"\r\n')

	while (1):
		line1 = ff1.readline()[0:-2]
		line2 = ff1.readline()[0:-2]
		line = line1 + "," + '\"'+ line2 + '\"\r\n'
		if (line1 == ''): 
			break
		ff2 = open(arff, "a+")
		ff2.write(line)        
	    ff2.close()

for name in filename:
	train = name + ".arff"
	test = "../test/"+ name + ".arff"
	at = name + "_att.arff"
	att = open(at, "a+")
	l1 = "@relation " + name + "\r\n"
	l2 = "@attribute class {false, true} \r\n" + "@attribute text string \r\n" + "@data\r\n"
	att.write(l1)
	att.write(l2)
	att.write(train)
	att.write(test)


for name in filename:
	train = name + ".arff"
	test = "../test/"+ name + ".arff"
	at = name + "_att.arff"
	att = open(at, "a+")
	l1 = "@relation " + name + "\r\n"
	l2 = "@attribute class {false, true} \r\n" + "@attribute text string \r\n" + "@data\r\n"
	att.write(l1)
	att.write(l2)
	att.write(train)
	att.write(test)






