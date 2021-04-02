import pandas as pd
pd.options.mode.chained_assignment = None
import sys
from sklearn.preprocessing import StandardScaler 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import joblib
import pickle
import random
import logging

logger = logging.getLogger(__file__)


class DataLoader():
	"""
	This class is responsible for all required data preprocessing before
		modeling occurs
	Constructor args:
		module: train/test
		datapath: path to data directory
	"""
	def __init__(self, args, config):
		self.datapath = args.datapath
		self.filepath = os.path.join(self.datapath, args.filename)
		with open(os.path.join(self.datapath, self.filepath), 'rb') as handle:
			self.embDict = pickle.load(handle)
		random.seed(0)

	def data_processing(self):
		"""
		Performs preprocessing and performs data split 
		"""	
		# Train/Eval/Test Split
		combiset = set(list(self.embDict.keys()))
		trainset = random.sample(combiset, int(0.80*len(combiset)))
		evalset = random.sample((combiset).difference(trainset), int(0.5*len((combiset).difference(trainset))))
		testset = combiset.difference(trainset).difference(evalset)
		return self.embDict, trainset, evalset, testset

