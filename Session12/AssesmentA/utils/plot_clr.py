import numpy as np
import matplotlib.pyplot as plt

def plot_clr():

	max_iterations=4000
	step_size=200



	lrmin=0.5
	lrmax=5

	lrt_trace=[]

	for iterations in range(1,max_iterations+1):
		cycle = np.floor(1+iterations/(2*step_size))
		x = np.abs(iterations/step_size - 2*cycle + 1)
		lrt = lrmin + (lrmax-lrmin)*np.maximum(0, (1-x))
	
		lrt_trace.append(lrt)


	fig= plt.figure(figsize=(15,4))
	plt.plot(range(1,max_iterations+1), lrt_trace)
	plt.show()