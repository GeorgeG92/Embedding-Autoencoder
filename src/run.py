from arg_parser import arg_parser
from config_parser import config_parser
from model import EmbeddingAutoEncoder
from data_loader import DataLoader
import sys
import logging



def main(args, config):
	loader = DataLoader(args, config)
	embDict, trainset, evalset, testset = loader.data_processing()
	model = EmbeddingAutoEncoder(args, config)
	model.run_pipeline(embDict, trainset, evalset, testset)


if __name__ == "__main__":
	args = arg_parser(sys.argv)
	config = config_parser(args)
	logging.basicConfig(level=getattr(logging, args.logging_level), 
		format="%(asctime)s|%(filename)-20.20s||%(funcName)-20.20s|%(levelname)-8.8s|%(message)s")
	main(args,config)
