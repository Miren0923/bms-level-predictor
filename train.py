import tensorflow as tf
import os
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
import libpatternParser as parser
import argparse

def argument_parser_helper():
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--network', help='load network(folder name)', type=str)
	parser.add_argument('-g', '--grad_num', type=int, default=0)
	return parser.parse_args()

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

period = 0.1
time = 20 # 0.05s * 40, 2s
key = 8

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

args = argument_parser_helper()

model = MyModel()
model._set_inputs(np.random.random((1, time, 8)))
if args.network:
	model.load_weights(os.path.join(os.getcwd(), 'network', args.network))

lr = 1e-3
optimizer = tf.keras.optimizers.Adam(lr=lr)

pattern_dir = os.path.join(os.getcwd(), 'satellitebms')
data = []
label = []

for pattern_name in os.listdir(pattern_dir):
	data.append(parser.patternParser(os.path.join(pattern_dir, pattern_name), period))
	label.append(int(pattern_name.split('|')[0]))

batch_size = 16
idx = np.arange(len(data))

train_summary_writer = tf.summary.create_file_writer(os.path.join(os.getcwd(), 'tensorboard_log'))
loss_name = ["loss", "loss_model", "loss_level", "loss_fail_success"]

grad_step = args.grad_num
for epoch in range(1000000):
	print("epoch : {}".format(epoch))
	np.random.shuffle(idx)
	for batch_idx in range(len(data) // batch_size):
		grad_step += 1
		print("batch_idx : {}".format(batch_idx))
		with tf.GradientTape() as tape:
			level_loss, fail_success_loss = tf.Variable(0.0), tf.Variable(0.0)
			for pattern_idx in range(batch_size):
				current_idx = idx[pattern_idx + batch_size * batch_idx]
				pattern_data = tf.convert_to_tensor([data[current_idx][i:i+time] for i in range(data[current_idx].shape[0] - time + 1)])
				label_data = label[current_idx]
				predict = model(pattern_data, training=True)
				hp = tf.ones(12)
				hp_min = tf.ones(12)
				predict_hp = tf.minimum((predict - 0.5) * 0.1, (predict - 0.5) * 0.002) * tf.reshape(tf.reduce_sum(pattern_data[:, time-1, :], axis=1), [-1, 1])
				predict_hp_low = tf.where(tf.greater(predict_hp, 0.0), predict_hp, predict_hp * 0.5)
				for i in range(predict.shape[0]):
					hp = hp + tf.where(tf.greater(hp, 0.3), predict_hp[i], predict_hp_low[i])
					hp = tf.minimum(hp, tf.ones(12))
					hp_min = tf.minimum(hp, hp_min)
				
				clip = tf.convert_to_tensor([1. for _ in range(label_data)] + [-1. for _ in range(12 - label_data)])
				fail_success_loss = fail_success_loss + tf.reduce_sum(tf.where(tf.greater(clip * hp_min, 0.0), clip * hp_min + 20.0, 0.0))
			level_loss = level_loss / batch_size
			fail_success_loss = fail_success_loss / batch_size
			model_loss = tf.reduce_sum(model.losses)
			loss = level_loss + fail_success_loss + model_loss

		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		with train_summary_writer.as_default():
			loss_detail = [loss, model_loss, level_loss, fail_success_loss]
			for i in range(len(loss_name)):
				tf.summary.scalar(loss_name[i], loss_detail[i], step=grad_step)
	
		if grad_step%1 == 0: model.save_weights(os.path.join(os.getcwd(), "network", "network_latest"))
		if grad_step%100 == 0 and grad_step > 0: model.save_weights(os.path.join(os.getcwd(), "network", "network_{}".format(grad_step)))

	lr *= 0.99
	optimizer.from_config({"lr" : lr})


