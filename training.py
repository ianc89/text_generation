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
training_data["1"] = training_data["1"].astype(str).str.lower()
training_data["2"] = training_data["2"].astype(str).str.lower()
# Attach trailing characters
training_data["0"] += ";"
training_data["1"] += ";"
training_data["2"] += "\n"

# Find a mapping for all characters
chars0 = sorted(list(set(" ".join(training_data['0'].values.flatten()))))
chars1 = sorted(list(set(" ".join(training_data['1'].values.flatten()))))
chars2 = sorted(list(set(" ".join(training_data['2'].values.flatten()))))
chars  = sorted(list(set(chars0+chars1+chars2)))

m = Mapping()
# prepare the dataset of input to output pairs encoded as integers
seq_length = 15
m.create(chars, seq_length)
m.save()

print (m.n_vocab)
print (m.chars_to_int)
# Training data - list of chars of seq_length
dataX_c  = []
# Target data   - single character (next character in sequence)
dataY    = []

# Loop through our data
for index, row in training_data.iterrows():
	row_data = row["0"]+row["1"]+row["2"]
	# Build our training and targets
	n_chars = len(row_data)
	for i in range(0, n_chars - seq_length, 1):
		seq_in  = row_data[i:i + seq_length]
		seq_out = row_data[i + seq_length]
		dataX_c.append([m.chars_to_int[char] for char in seq_in])
		dataY.append(m.chars_to_int[seq_out])
		if index == 0:
			print (seq_in,"->",seq_out)

# Number of pattens in training data
n_patterns = len(dataX_c)
# reshape X to be [samples, time steps, features] 
X = numpy.reshape(dataX_c, (n_patterns, seq_length, 1))
# normalize
X = X / float(m.n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

mo = ModelClass()
mo.set_model_4(X, y)
model = mo.model
print (model.summary())
plot_model(model, to_file='rnn.png')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-model4.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
# Model 1
#model.fit([X_s, X], y, epochs=10, batch_size=128, callbacks=callbacks_list)
# Model 2
#model.fit([X_s2, X], y, epochs=5, batch_size=128, callbacks=callbacks_list)
# Model 3
model.fit(X, y, epochs=3, batch_size=256, callbacks=callbacks_list)


