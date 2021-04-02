import os
import yaml


def config_parser(args):
	"""Open .yaml configuration file and parse the content
	Args:
		args: an arguments object containing config path and name
	Returns:
		An dictionary object that contains config parameters
	"""
	# Creating a parser
	with open(os.path.join(args.configpath, args.configname)) as file:
		config = yaml.load(file, Loader=yaml.FullLoader)
	return config

