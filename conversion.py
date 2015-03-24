filename = raw_input()
fn1 = filename + "\.train"
fn2 = filename + "\.arff"
f1 = open(fn1, "wr+")
while (1):
	line = f1.readline()[0:-2] + "," + f1.readline()
	f2 = open(fn2, "wr+")
	f2.read()
    f2.write(line)        
    f2.close()



