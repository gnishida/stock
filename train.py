import datetime
import os
from PIL import Image, ImageDraw
import numpy
import scipy.misc
import glob
import scipy
import random
import argparse
import cv2

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


MODEL_FILE_NAME = "nn_model.h5"


def load_data(filepath):
	X = []
	Y = []
	file = open(filepath, "r")
	for line in file.readlines():
		data = line.split(",")
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
		tf.keras.layers.Dropout(0.25),
		tf.keras.layers.LSTM(256),
		tf.keras.layers.Dropout(0.25),
		tf.keras.layers.Dense(64, activation='relu'),
		tf.keras.layers.Dense(1, name='fc')
	])

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	model.compile(loss='mse',
		optimizer=optimizer,
		metrics=['mae', 'mse'])
	return model
  

def train(filepath, model_dir, num_epochs, learning_late):
	# Load data
	X, Y = load_data(filepath)
	print(X.shape)

	# Build model
	model = build_model(X[0].shape, learning_late)

	# Setup for Tensorboard
	log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
	file_writer.set_as_default()
	tensorboard_callback = TensorBoard(
		log_dir=log_dir,
		update_freq='batch',
		histogram_freq=1)

	# Training model
	model.fit(X, Y,
		epochs=num_epochs,
		validation_split = 0.2,
		callbacks=[tensorboard_callback])

	# Save the model
	model.save("{}/{}".format(model_dir, MODEL_FILE_NAME))


def test(filepath, model_dir):
	# Load data
	X, Y = load_data(filepath)
		  
	# Load the model
	model = tf.keras.models.load_model("{}/{}".format(model_dir, MODEL_FILE_NAME))
		
	# Evaluation
	model.evaluate(X, Y)
	
	# Prediction
	predictedY = model.predict(X).flatten()


def main():	
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', required=True, choices=["train", "test"])
	parser.add_argument('--filepath', required=True, help="path to the data file")
	parser.add_argument('--model_dir', default="models", help="path to folder containing models")
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--learning_rate', type=float, default=0.0001)
	args = parser.parse_args()	

	if args.mode == "train":
		train(args.filepath, args.model_dir, args.num_epochs, args.learning_rate)
	elif args.mode == "test":
		test(args.filepath, args.model_dir)
	else:
		print("Invalid mode is specified {}".format(args.mode))
		exit(1)
	

if __name__== "__main__":
	main()
