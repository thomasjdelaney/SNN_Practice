import argparse, os, sys, shutil, h5py
import pyNN.nest as pynn
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from PlottingFunctions import *

# eventually this script will evaluate the performance of the network
parser = argparse.ArgumentParser(description='Load in the results of training, and have a look.')
parser.add_argument('-f', '--file_path_name', help='The h5 file for loading.', default=os.path.join(os.environ['HOME'],'SNN_practice','h5','training_results.h5'), type=str)
parser.add_argument('-s', '--numpy_seed', help='For seeding random numbers', default=1798, type=int)
parser.add_argument('--debug', help='enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

np.random.seed(args.numpy_seed)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

proj_dir = os.path.join(os.environ['HOME'], 'SNN_practice')
h5_dir = os.path.join(proj_dir, 'h5')

h5_file = h5py.File(args.file_path_name, 'r')
