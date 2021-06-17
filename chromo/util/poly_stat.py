"""Average Polymer Statistics

Generate average polymer statistics from Monte Carlo simulation output. This
module defines a PolyStat object, which reads polymer configurations from an
output file, samples beads, and calculates basic polymer statistics, such as:
mean squared end-to-end distance, mean squared radius of gyration, and fourth 
moment end-to-end distance.

Bead sampling methods include overlapping sliding windows and non-overlapping
sliding windows. Overlapping sliding windows offer the benefit of increased
data, though the results are biased by central beads which exist in multiple 
bins of the average. Non-overlapping sliding windows reduce the bias in the
results, but include fewer samples in the average.

Joseph Wakim
Spakowitz Lab
November 1, 2020

Usage: python poly_stat.py --path <DATA PATH> --sampling <SAMPLING METHOD>

"""

import os
import csv
import sys
import argparse

import pandas as pd
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def read_arguments():
	"""
	Read input arguments from the command line when module is called.

	This function allows for data paths and sampling methods to be 
	specified in the command line when this module is called as __main__
	in terminal.

	Create an argument parser object, and specify argument flags. Then, parse
	the arguments from the command line.

	Returns
	-------
	args : Namespace
		Input arguments from python module call in terminal
	"""
	arg_parser = argparse.ArgumentParser(description="Read inputs.")
	arg_parser.add_argument("--path", type=str, help="Path to MC simulation \
		output file")
	arg_parser.add_argument("--sampling", type=str, help="Bead sampling \
		method, either 'overlap_slide' or 'jump_slide'")
	args = arg_parser.parse_args()
	return args


class PolyStat(object):
	"""Generate polymer statistics to summarize polymer configuration."""

	def __init__(self, file_path):
		"""
		Initialize PolyStat object.

		Parameters
		----------
		file_path : str
			Path from main directory to file with polymer configuration
 	 	"""

		# Path to polymer configuration file
		self.file_path = file_path
		# Bead positions and orientations
		self.poly_config = None
		# Number of beads in polymer
		self.num_beads = None
		# Average squared end-to-end distance of polymer
		self.avg_r2 = None
		# Load polymer configuration data
		self.load_data()

	def load_data(self):
		"""Load polymer configuration from result file.
		
		Load polymer configuration data from a CSV file and store the polymer
		configuration as an object attribute.
		"""
		data_table = pd.read_csv("../../" + self.file_path, skiprows=1)
		num_beads = len(data_table.index)
		self.poly_config = data_table
		self.num_beads = num_beads

	def get_avg_r2(self, sampling_scheme, bead_separation):
		"""
		Calculate the average squared end-to-end distance of polymer.

		Parameters
		----------
		sampling_scheme : str
			Polymer bead sampling scheme for calculation of average distances.
			Must be one of the following: "overlap_slide" for a sliding window
			average with overlapping bins, "jump_slide" for a sliding window
			average with mutually exclusive bins.
		bead_separation : int
			Number of beads in window for average calculation

		Returns
		-------
		avg_r2 : float
			Average squared end-to-end distance
		"""
		windows = self.sample_beads(sampling_scheme, bead_separation)
		avg_r2 = self.calc_r2_avg(windows)
		return avg_r2

	def get_avg_r4(self, sampling_scheme, bead_separation):
		"""
		Calculate the average fourth moment end-to-end distance of polymer.

		Parameters
		----------
		sampling_scheme : str
			Polymer bead sampling scheme for calculation of average distances.
			Must be one of the following: "overlap_slide" for a sliding window
			average with overlapping bins, "jump_slide" for a sliding window
			average with mutually exclusive bins.
		bead_separation : int
			Number of beads in window for average calculation

		Returns
		-------
		avg_r4 : float
			Average fourth moment end-to-end distance
		"""
		self.sample_beads(sampling_scheme)
		avg_r4 = self.calc_r4_avg(bead_separation)
		return avg_r4

	def overlapping_slide_sample(self, bead_separation):
		"""
		Make list of index pairs for sliding window sampling scheme.

		Parameters
		----------
		bead_separation : int
			Number of beads in window for average calculation

		Returns
		-------
		windows : array_like (N, 2)
			Pairs of bead indicies for windows of statistics
		"""
		num_windows = self.num_beads - bead_separation
		windows = np.zeros((num_windows, 2))
		for i in range(num_windows):
			windows[i, 0] = i
			windows[i, 1] = i + bead_separation
		windows = windows.astype("int64")
		return windows

	def jumping_slide_sample(self, bead_separation):
		"""
		Make list of index pairs for mutually exclusive bin sampling scheme.

		If end of polymer does not constitute a complete bin, it is excluded
		from the average. 

		Parameters
		----------
		bead:separation : int
			Number of beads in window for average calculation

		Returns
		-------
		windows : array_like (N, 2)
			Pairs of bead indicies for windows of statistics
		"""
		num_windows = int(np.floor(self.num_beads / bead_separation))
		windows = np.zeros((num_windows, 2))
		for i in range(num_windows):
			bin_start = i * bead_separation
			windows[i, 0] = bin_start
			windows[i, 1] = bin_start + bead_separation - 1
		windows = windows.astype("int64")
		return windows

	def sample_beads(self, sampling_scheme, bead_separation):
		"""
		Sample beads based on specified sampling scheme.

		Parameters
		----------
		sampling_scheme : str
			Polymer bead sampling scheme for calculation of average distances.
			Must be one of the following: "overlap_slide" for a sliding window
			average with overlapping bins, "jump_slide" for a sliding window
			average with mutually exclusive bins.
		bead_separation : int
			Number of beads in window for average calculation
		
		Returns
		-------
		array_like (N, 2)
			List of start and end indicies for window averaging.
		"""
		if sampling_scheme == "overlap_slide":
			return self.overlapping_slide_sample(bead_separation)
		elif sampling_scheme == "jump_slide":
			return self.jumping_slide_sample(bead_separation)
		else:
			raise ValueError("Bead sampling scheme not recognized.")

	def get_starting_and_ending(self, windows):
		"""
		Generate arrays of starting and ending x, y, z coordinates in windows.

		Begins by identifying number of pairs in the average. Then initializes
		vectors of starting and ending coordinates. Lastly, fills in the 
		starting and ending coordinates.
		
		Parameters
		----------
		windows : array_like (N, 2)
			Pairs of bead indicies for windows of statistics

		Returns
		-------
		start : array_like (N, 3)
			x, y, z coordinates at starting index of window
		end : array_like (N, 3)
			x, y, z coordinate at ending index of window
		"""
		num_pairs = len(windows)
		start = np.zeros((num_pairs, 3))
		end = np.zeros((num_pairs, 3))

		for i in range(len(windows)):
			start[i, 0] = self.poly_config.iloc[windows[i,0], 1]
			start[i, 1] = self.poly_config.iloc[windows[i,0], 2]
			start[i, 2] = self.poly_config.iloc[windows[i,0], 3]
			end[i, 0] = self.poly_config.iloc[windows[i,1], 1]
			end[i, 1] = self.poly_config.iloc[windows[i,1], 2]
			end[i, 2] = self.poly_config.iloc[windows[i,1], 3]

		return start, end

	def calc_r2_avg(self, windows):
		"""
		Calculate the avg squared end-to-end distance for given bead pairs.

		Parameters
		----------
		windows : array_like (N, 2)
			Pairs of bead indicies for windows of statistics

		Returns
		-------
		avg_r2 : float
			Average squared end-to-end distance
		"""
		start, end = self.get_starting_and_ending(windows)
		avg_r2 = np.mean(np.sum(np.square(end-start), axis=1))
		return avg_r2

	def calc_r4_avg(self, windows):
		"""
		Calculate the avg fourth moment end-to-end distance for bead pairs.

		Parameters
		----------
		windows : array_like (N, 2)
			Pairs of bead indicies for windows of statistics

		Returns
		-------
		avg_r4 : float
			Average fourth moment end-to-end distance
		"""
		start, end = self.get_starting_and_ending(windows)
		avg_r4 = np.mean(np.sum(np.power(end-start, 4), axis=1))
		return avg_r4


def main():
	""" Run the program."""

	# Read arguments from module call
	args = read_arguments()
	
	# Specify data path and bead sampling method
	data_path = args.path
	sampling = args.sampling
	
	# Create the PolyStat object
	poly_stat = PolyStat(data_path)

	avg_r2_vec = []
	for i in range(3, 501):
		avg_r2 = poly_stat.get_avg_r2("jump_slide", i)
		avg_r2_vec.append(avg_r2)
	print(avg_r2_vec)


if __name__ == "__main__":
	main()

