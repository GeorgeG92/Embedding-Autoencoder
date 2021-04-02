import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR         	# suppress tensorflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)   	# run before importing tensorflow

import tensorflow.compat.v1 as tf           				# required to use tfv1 code in tfv2, 
tf.disable_v2_behavior()
# import tensorflow as tf                   				# uncomment if code is written in tfv2

from threading import Thread
from queue import Queue
import joblib
import numpy as np
import random

logger = logging.getLogger(__file__)


class EmbeddingAutoEncoder():
	"""
	This class wraps the Tensorflow model that is used for the project
	Constructor args:
		args: argument class as passed by the argument parser
	"""
	def __init__(self, args, config):
		self.modelpath = args.modelpath+'/'
		self.modelname = args.modelname
		self.mode = args.mode
		self.batchsize = config['model']['batch_size']    
		self.learningrate = config['model']['learning_rate']
		self.beta1 = config['model']['beta_1']                 
		self.batchesNo = config['model']['batches_num']
		self.dropoutrate = config['model']['dropout_rate']
		self.inputsize = config['model']['input_size']
		self.outputsize = self.inputsize
		self.queue = Queue(maxsize=10)
		self.feature_dict = {}               			# dictionary of features
		self.target_dict = {}                			# dictionary of targets
		self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), log_device_placement=False,
			allow_soft_placement=True)
		self.check_device()

	def mae_loss(self, outputs, targets):
		""" Mean Absolute Error loss for embedding reconstruction
		Args:
			targets: the ground truth passed to the model
			predictions: the output of the model
		Returns:
			Loss: the mean absolute error loss
		"""
		loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(targets-outputs),1),0)
		return loss

	def fc_layer(self, x, w, b, activation=None):
		""" Implementation of a fully connected/dense TF layer
		Args:
			x: the input of the layer
			w: the corresponding weight matrix
			b: the bias vector
			activation: the activation function to be applied in the end
		Returns:
			The result of the computation
		"""
		result = tf.matmul(x, w) + b
		if activation == 'relu':
			result = tf.nn.leaky_relu(result)         # normal will cut off most values
		if activation == 'softmax':
			result = tf.nn.softmax(result,axis=1)
		return result

	def check_device(self):
		""" Checks if GPU is available to tensorflow and prints a message
		"""
		if len(tf.config.experimental.list_physical_devices('GPU'))>0:
			logger.info("Model Loading with GPU support")
		else:
			logger.info("Model Loading on CPU")

	def save_model(self, step):
		""" Saves trained model to disk
		"""
		#self.saver.save(sess, self.modelpath)
		#tf.saved_model.save(tf.trainable_variables(), self.modelpath)
		logger.info("Saving model at {path}".format(path=os.path.join(self.modelpath, self.modelname)))
		saved_path = self.saver.save(self.sess, 
			save_path=os.path.join(self.modelpath, self.modelname),
			global_step=step)

	def load_model(self):
		""" Loads latest model graph metadata and weights from disk
		"""
		logger.info("Loading model from {path}".format(path=os.path.join(self.modelpath)))
		#print("Graph:",os.path.join(self.modelpath, 'george-'+str(self.batchesNo)+'.meta'))
		self.saver = tf.train.Saver()
		self.saver = tf.train.import_meta_graph(os.path.join(self.modelpath, 'george-'+str(self.batchesNo)+'.meta'))
		self.saver.restore(self.sess, tf.train.latest_checkpoint(self.modelpath))

	def batch_prep(self):
		""" The code of batch preparation: runs from a separate thread during training
		"""
		for i in range(self.batchesNo):
			drawIds = random.sample(self.trainset, self.batchsize)
			contFeatures = np.array([self.feature_dict[x] for x in drawIds], dtype=object)
			targets = np.array([self.target_dict[x] for x in drawIds], dtype=object)
			self.queue.put((contFeatures, targets))

	def prepare_data(self):
		""" Prepares the dictionaries that will be used during batch prep
			and defined train, eval and test examples
		"""
		logger.info("\tCreating dictionaries for batch_prep")
		self.feature_dict = {}
		self.target_dict = {}
		for data in [self.trainset, self.evalset, self.testset]:
			for matCombi in data:    
				self.feature_dict[matCombi] = self.embDict[matCombi][1]                    
				self.target_dict[matCombi] = self.embDict[matCombi][1]           

	def init_weights_and_biases(self):
		""" Initializes the trainable parameters of the model
		"""
		self.biases = {
			'b1': tf.get_variable('b1', shape=(768,), initializer=tf.random_normal_initializer(0, 0.05)),
			'b2': tf.get_variable('b2', shape=(512,), initializer=tf.random_normal_initializer(0, 0.05)),
			'b3': tf.get_variable('b3', shape=(256,), initializer=tf.random_normal_initializer(0, 0.05)),
			'b4': tf.get_variable('b4', shape=(512,), initializer=tf.random_normal_initializer(0, 0.05)),
			'b5': tf.get_variable('b5', shape=(768,), initializer=tf.random_normal_initializer(0, 0.05)),
			'b6': tf.get_variable('b6', shape=(self.outputsize,), initializer=tf.random_normal_initializer(0, 0.05))
		}
		self.weights = {
			'w1': tf.get_variable('w1', shape=(self.inputsize, 768), initializer=tf.random_normal_initializer(0, 0.05)),
			'w2': tf.get_variable('w2', shape=(768, 512), initializer=tf.random_normal_initializer(0, 0.05)),
			'w3': tf.get_variable('w3', shape=(512, 256), initializer=tf.random_normal_initializer(0, 0.05)),
			'w4': tf.get_variable('w4', shape=(256, 512), initializer=tf.random_normal_initializer(0, 0.05)),
			'w5': tf.get_variable('w5', shape=(512, 768), initializer=tf.random_normal_initializer(0, 0.05)),
			'w6': tf.get_variable('w6', shape=(768, self.outputsize), initializer=tf.random_normal_initializer(0, 0.05))
		}

	def define_model_architecture(self):
		""" Defines the model architecture: the input placeholders and the operations within the graph
		"""
		self.features_input = tf.placeholder(name="featuresInput", dtype=tf.float32, shape=(None, self.inputsize))
		self.dropoutprob = tf.placeholder_with_default(0.3, shape=())
		self.targets_output = tf.placeholder(name="targetsOutput", dtype=tf.float32, shape=(None, self.outputsize))

		with tf.variable_scope("contFeatures"):
			self.x1 = self.fc_layer(self.features_input, self.weights['w1'], self.biases['b1'], 'relu')
			self.x1 = tf.nn.dropout(self.x1, rate=self.dropoutprob)
			self.x2 = self.fc_layer(self.x1, self.weights['w2'], self.biases['b2'], 'relu')
			self.x2 = tf.nn.dropout(self.x2, rate=self.dropoutprob)
				
			self.x3 = self.fc_layer(self.x2, self.weights['w3'], self.biases['b3'], 'relu')            # embedding of interest
			self.x3 = tf.nn.dropout(self.x3, rate=self.dropoutprob)
			   
			self.x4 = self.fc_layer(self.x3, self.weights['w4'], self.biases['b4'], 'relu')
			self.x4 = tf.nn.dropout(self.x4, rate=self.dropoutprob)
			self.x5 = self.fc_layer(self.x4, self.weights['w5'], self.biases['b5'], 'relu')
			self.x5 = tf.nn.dropout(self.x5, rate=self.dropoutprob)
			self.model_output = self.fc_layer(self.x5, self.weights['w6'], self.biases['b6'])
	
		self.loss = self.mae_loss(self.model_output, self.targets_output)
		self.optim = tf.train.AdamOptimizer(self.learningrate, self.beta1)                          
		self.train = self.optim.minimize(self.loss)

	def run_inference(self):
		""" Used to run model on eval/test data and compute appropriate metrics
		"""
		self.prepare_data()
		self.init_weights_and_biases()
		self.define_model_architecture()
		logger.info("\tTesting")
		with tf.Session(config=self.config) as self.sess:
			self.sess.run(tf.global_variables_initializer())
			self.load_model()

			# Prepare Graph inputs and process outputs
			cont_features = np.array([self.feature_dict[x] for x in self.testset], dtype=object)
			targets = np.array([self.target_dict[x] for x in self.testset], dtype=object)
			fd = {self.features_input: cont_features, self.targets_output: targets, self.dropoutprob: 0.6}
			_, l, o = self.sess.run(fetches=[self.train, self.loss, self.model_output], feed_dict=fd)

			# Calculate test loss
			logger.info(f"\tTest loss: {round(l, 5)}")

	def run_evaluation(self, sess):
		""" Used to run model on eval/test data and compute appropriate metrics
		Args:
			sess: The TF session passed from the train function for evaluation
		"""
		logger.info("Evaluating:")
		cont_features = np.array([self.feature_dict[x] for x in self.evalset], dtype=object)
		targets = np.array([self.target_dict[x] for x in self.evalset], dtype=object)
		fd = {self.features_input: cont_features, self.targets_output: targets, self.dropoutprob: 0.6}
		_, l, o = self.sess.run(fetches=[self.train, self.loss, self.model_output], feed_dict=fd)
		logger.info(f"\tEvaluation loss: {round(l, 5)}")

	def init_thread(self):
		"""
		Initializes the batch prep thread
		"""
		t = Thread(target=self.batch_prep)
		t.daemon = True
		t.start()

	def run_pipeline(self, embDict, trainset, evalset, testset):
		"""
		Runs the model pipeline depending on user arguments
		"""
		self.embDict, self.trainset, self.evalset, self.testset = embDict, trainset, evalset, testset
		self.prepare_data()
		self.init_weights_and_biases()
		self.define_model_architecture()
		self.train_model()

	def train_model(self):
		"""
		Initializes a TF session, performs training and predicts on eval/test data
		Returns:
			Success Code
		"""
		with tf.Session(config=self.config) as self.sess:
			self.sess.run(tf.global_variables_initializer())
			self.saver = tf.train.Saver()
			self.init_thread()
			#print("Training:")
			logger.info("Training:")
			self.loss_dict = {}
			self.eval_loss_dict = {}
			for i in range(self.batchesNo):
				batch_tuple = self.queue.get()
				cont_features, targets = batch_tuple[0], batch_tuple[1]
				fd = {self.features_input: cont_features, self.targets_output: targets, self.dropoutprob: self.dropoutrate}
				_, l, o = self.sess.run(fetches=[self.train, self.loss, self.model_output], feed_dict=fd)
				self.loss_dict[i] = l
				if i % (int(self.batchesNo / 50)) == 0:
					avg_train_loss = np.array(list(self.loss_dict.values()))[-2000:].mean()
					logger.info(f"\tBatch {i}: Average train loss ( last {min(i,2000)} batches): {avg_train_loss}")
				if self.evalset is not None:
					if i > 0 and i % (int(self.batchesNo / 10)) == 0:
						self.run_evaluation(sess=self.sess)
				self.queue.task_done()
			try:
				self.queue.join()                    # Join batch thread
			except KeyboardInterrupt:
				sys.exit(1)
			self.save_model(self.batchesNo)