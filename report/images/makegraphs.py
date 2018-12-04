import sys
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def run():
	index = range(100,2100,100)

	win_rate=[0.0, 0.0, 0.26, 0.4, 0.425, 0.41, 0.455, 0.575, 0.54, 0.69, 0.725, 0.965, 0.965, 0.955, 0.985, 0.96, 0.975, 1.0, 0.96, 0.985]

	plt.figure()
	plt.plot(index, win_rate)
	plt.title("Win Rate vs Num. Training Rounds")
	plt.xlabel("Number of Training Rounds")
	plt.ylabel("Win Rate")

	plt.show()

if __name__ == "__main__":
	run()
