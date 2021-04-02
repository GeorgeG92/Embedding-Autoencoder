import argparse
import os
#import yaml


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def arg_parser(argv):
	"""Add command line arguments and parse user inputs.
	Args:
		argv: User input from the command line.
	Returns:
		An args object that contains parsed arguments.
	"""
	# Creating a parser
	print("Argument Parser...")
	parser = argparse.ArgumentParser(description='MyInstance')
	parser.add_argument('-l', '--logging', dest='logging_level', type=str.upper,
						choices=['debug', 'info', 'warning', 'error', 'critical'],
						default='info', help='set logging level')

	parser.add_argument('-m', '--mode', type=str.lower, choices=['train', 'inference'],
						default='inference', help='Train/Inference selection')

	parser.add_argument('--configpath', type=str, dest='configpath',
						default='../config', help='Config directory')

	parser.add_argument('--configname', type=str, dest='configname',
						default='config.yaml', help='Config file name')

	parser.add_argument('--modelpath', type=str, dest='modelpath',
						default='../model', help='Model directory')

	parser.add_argument('--modelname', type=str, dest='modelname',
						default='george_ae', help='Model Name')

	parser.add_argument('--datapath', type=str, dest='datapath',
						default='../data', help='Data directory')

	parser.add_argument('--filename', type=str, dest='filename',
						default='embDict.pickle', help='Name of the embeddings dictionary')


	args = parser.parse_args(argv[1:])
	assert os.path.exists(args.datapath), f"data directory not found at {args.datapath}"
	assert os.path.exists(os.path.join(args.datapath, args.filename)), f"data file not found at {os.path.join(args.datapath, args.filename)}"
	assert os.path.exists(os.path.join(args.configpath, args.configname)), f"config file not found at {os.path.join(args.datapath, args.filename)}"
	if not os.path.exists(args.modelpath):
		os.makedirs(args.modelpath)
	return args

