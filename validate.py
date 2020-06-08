import tensorflow as tf
import os
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
import libpatternParser as parser
import argparse

def argument_parser_helper():
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--network', help='load network', type=str)
	return parser.parse_args()

class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv2a = Conv2D(4, (5, 1)) # 20 * 8 * 1 -> 16 * 8 * 4
		self.bn2a = BatchNormalization()
		self.conv2b = Conv2D(8, (5, 4)) # 16 * 8 * 4 -> 12 * 5 * 8
		self.bn2b = BatchNormalization()
		self.conv2c = Conv2D(16, (5, 4)) # 12 * 5 * 8 -> 8 * 2 * 16
		self.bn2c = BatchNormalization()
		self.flatten = Flatten()
		self.d1 = Dense(13, activation='sigmoid')

	def call(self, x, training=False):
		x = tf.reshape(x, [-1, time, 8, 1])
		x = self.conv2a(x)
		x = self.bn2a(x, training=training)
		x = tf.nn.relu(x)

		x = self.conv2b(x)
		x = self.bn2b(x, training=training)
		x = tf.nn.relu(x)

		x = self.conv2c(x)
		x = self.bn2c(x, training=training)
		x = tf.nn.relu(x)

		x = self.flatten(x)
		x = self.d1(x)
		x = tf.math.cumsum(x, axis=1) / tf.math.reduce_sum(x, axis=1, keepdims=True)
		return x[:, 0:12]

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

pattern_dir = os.path.join(os.getcwd(), 'satellitebms')

period = 0.1
time = 20 # 0.05s * 40, 2s
key = 8

args = argument_parser_helper()
model = MyModel()
model._set_inputs(np.random.random((1, time, 8)))
if args.network:
	model.load_weights(os.path.join(os.getcwd(), args.network))

for pattern_name in os.listdir(pattern_dir):
	data = parser.patternParser(os.path.join(pattern_dir, pattern_name), period)
	label = int(pattern_name.split('|')[0])

	pattern_data = tf.convert_to_tensor([data[i:i+time] for i in range(data.shape[0] - time + 1)])
	predict = model(pattern_data, training=True)
	predict_hp = tf.minimum((predict - 0.5) * 0.1, (predict - 0.5) * 0.002) * tf.reshape(tf.reduce_sum(pattern_data[:, time-1, :], axis=1), [-1, 1])
	predict_hp_low = tf.where(tf.greater(predict_hp, 0.0), predict_hp, predict_hp * 0.5)
	hp = tf.ones(12)
	hp_min = tf.ones(12)
	for i in range(predict.shape[0]):
		hp = hp + tf.where(tf.greater(hp, 0.3), predict_hp[i], predict_hp_low[i])
		hp = tf.minimum(hp, tf.ones(12))
		hp_min = tf.minimum(hp, hp_min)
	
	lev = 12
	for i in range(12):
		if hp_min[i] > 0:
			lev = i
			break
	
	clip = tf.convert_to_tensor([1. for _ in range(label)] + [-1. for _ in range(12 - label)])
	fail_success_loss = tf.reduce_sum(tf.where(tf.greater(clip * hp_min, 0.0), clip * hp_min + 20.0, 0.0))
	print("pattern name : {}, predicted level : {}".format(pattern_name, lev))
	print("fail_success_loss : {}".format(fail_success_loss))
