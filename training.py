#!/usr/bin/env python3
# Using https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
import numpy
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, plot_model
from settings import Mapping
from models import ModelClass

# Load data
training_data = pd.read_csv("data/haikus1.csv", 
							names = ["0","1","2",
									 "source",
									 "0_syllables","1_syllables","2_syllables"],
							header=0)
# Attempt to clean up all non-alphanumeric characters
training_data.replace("[^a-zA-Z\d\s:]","",regex=True,inplace=True)
# Explict strings (move all to lower case too)
training_data["0"] = training_data["0"].astype(str).str.lower()
# Attach trailing characters
#training_data["0"] += ";"
# Check
print (training_data["0"][0])
# Find max length of 99th quantile
max_line_length    = int(max([training_data['0'].str.len().quantile(.99)]))
# Select only data within this range
training_data      = training_data[(training_data['0'].str.len() <= max_line_length)].copy()
# Find a mapping for all characters
chars = sorted(list(set(" ".join(training_data['0'].values.flatten()))))



m = Mapping()
# prepare the dataset of input to output pairs encoded as integers
seq_length = 10
m.create(chars, seq_length)
m.save()

print (m.n_vocab)
print (m.chars_to_int)
# Training data - list of chars of seq_length
dataX_c  = []
# Training data - single syllables in line
dataX_s  = []
# Training data - list of syllables in line of seq_length
dataX_s2 = []
# Target data   - single character (next character in sequence)
dataY    = []

# Loop through our data
for index, row in training_data.iterrows():
	# Build our training and targets
	n_chars = len(row["0"])
	for i in range(0, n_chars - seq_length, 1):
		seq_in  = row["0"][i:i + seq_length]
		seq_out = row["0"][i + seq_length]
		dataX_c.append([m.chars_to_int[char] for char in seq_in])
		dataX_s.append(row["0_syllables"])
		dataX_s2.append([row["0_syllables"] for char in seq_in])
		dataY.append(m.chars_to_int[seq_out])
		if index == 0:
			print (seq_in)

# Number of pattens in training data
n_patterns = len(dataX_c)
# reshape X to be [samples, time steps, features] 
X = numpy.reshape(dataX_c, (n_patterns, seq_length, 1))
# need to match for dataX_s as well, but sequence length is 1
X_s = numpy.reshape(dataX_s, (n_patterns, 1, 1))
# Try to shape into same batching
X_s2 = numpy.reshape(dataX_s2, (n_patterns, seq_length, 1))
# normalize
X = X / float(m.n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

mo = ModelClass()
mo.set_model_3(X, y)
model = mo.model
print (model.summary())
plot_model(model, to_file='rnn.png')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-model3.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
# Model 1
#model.fit([X_s, X], y, epochs=10, batch_size=128, callbacks=callbacks_list)
# Model 2
#model.fit([X_s2, X], y, epochs=5, batch_size=128, callbacks=callbacks_list)
# Model 3
model.fit([X_s, X], y, epochs=5, batch_size=256, callbacks=callbacks_list)


