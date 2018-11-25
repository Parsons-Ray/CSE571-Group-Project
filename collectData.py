import sys
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def run():
	agent = sys.argv[-1]
	win_rate = []
	index = range(50,2050,50)

	for i in index:
		training = i
		validation = training + 20

		command = "python pacman.py -p " + agent + " -x " + str(training) + " -n " + str(validation) + " -l smallGrid" 
		output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
		output = output.split("\n")

		win_rate.append(float(output[-3].split()[-1][1:-1]))

	plt.figure()
	plt.plot(index, win_rate)
	plt.title("Win Rate vs Num. Training Rounds")
	plt.xlabel("Number of Training Rounds")
	plt.ylabel("Win Rate")

	plt.show()

if __name__ == "__main__":
	run()
