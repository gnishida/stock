import argparse
import datetime
import numpy
import random
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


MODEL_FILE_NAME = "lstm_model.h5"


def load_data(filepath):
	X = []
	Y = []
	file = open(filepath, "r")
	for line in file.readlines():
		data = [float(x) for x in line.split(",")]
		X.append(data[:-1])
		Y.append(data[-1])

	TX = numpy.zeros((len(X), 1, len(X[0])), dtype=float)
	TY = numpy.zeros((len(X)), dtype=float)

	for i in range(len(X)):
		TX[i, 0, :] = X[i]
		TY[i] = Y[i]
	return TX, TY


def build_model(input_shape, learning_rate):
	model = tf.keras.Sequential([
		tf.keras.layers.GRU(256, input_shape=input_shape, return_sequences=True, name="GRU"),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.LSTM(256),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(64, activation="relu"),
		tf.keras.layers.Dense(1, name="fc")
	])

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	model.compile(loss="mean_squared_error",
		optimizer=optimizer,
		metrics=["mean_squared_error"])
	return model
  

def train(filepath, model_dir, num_epochs, learning_late):
	# Load data
	X, Y = load_data(filepath)

	# Build model
	model = build_model(X[0].shape, learning_late)

	# Setup for Tensorboard
	log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
	file_writer.set_as_default()
	tensorboard_callback = TensorBoard(
		log_dir=log_dir,
		update_freq="batch",
		histogram_freq=1)

	# Training model
	model.fit(X, Y,
		epochs=num_epochs,
		validation_split = 0.2,
		callbacks=[tensorboard_callback])

	# Save the model
	model.save("{}/{}".format(model_dir, MODEL_FILE_NAME))


def main():	
	parser = argparse.ArgumentParser()
	parser.add_argument("--filepath", default="train_data.txt", help="Path to the training data file")
	parser.add_argument("--model-dir", default="models", help="Path to folder containing models")
	parser.add_argument("--num-epochs", type=int, default=70, help="Number of epochs to train")
	parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
	args = parser.parse_args()	

	train(args.filepath, args.model_dir, args.num_epochs, args.learning_rate)
	

if __name__== "__main__":
	main()
