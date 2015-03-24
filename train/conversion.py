filename = ["acq", "corn", "earn", "interest", "ship", "crude", "trade", "grain", "money-fx", "wheat"]

for name in filename:
	train = name + ".train"
	arff = name + ".arff"
	ff1 = open(train, "r")
	ff2 = open(arff, "a+")
	ff2.write("@relation " + name+ '\r\n')
	ff2.write("@attribute class {true, false}"+ '\r\n')
	ff2.write("@attribute text String"+ '\r\n')
	ff2.write("@data"+ '\r\n')
	ff2.close()
	while (1):
		line1 = ff1.readline()[0:-2]
		line2 = ff1.readline()[0:-2]
		line = line1 + "," + '\"'+ line2 + '\"\r\n'
		if (line1 == ''): 
			break
		ff2 = open(arff, "a+")
		ff2.write(line)        
		ff2.close()