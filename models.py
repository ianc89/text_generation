from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Concatenate
from keras.layers import Reshape
from keras.layers import Add
from keras.layers import Dropout
class ModelClass:
	
	def __init__(self):
		self.model = None

	def set_model_1(self,X,y):
		input1 = Input(shape=(1,))
		input2 = Input(shape=(X.shape[1],X.shape[2]))
		lstm1  = LSTM(256)(input2)
		merged = Concatenate(axis=1)([input1, lstm1])
		output = Dense(y.shape[1], activation='softmax')(merged)
		model  = Model(inputs=[input1, input2], outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model = model

	def set_model_2(self,X,X_s2,y):
		input1 = Input(shape=(X_s2.shape[1],X_s2.shape[2]))
		input2 = Input(shape=(X.shape[1],X.shape[2]))
		merged = Add()([input1, input2])
		lstm1  = LSTM(512)(merged)
		output = Dense(y.shape[1], activation='softmax')(lstm1)
		model  = Model(inputs=[input1, input2], outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model = model
		
	def set_model_3(self,X,y):
		input1 = Input(shape=(1,))
		input2 = Input(shape=(X.shape[1],X.shape[2]))
		lstm1  = LSTM(1500)(input2)#512
		drop1  = Dropout(0.1)(lstm1)
		merged = Concatenate(axis=1)([input1, drop1])
		dense1 = Dense(100, activation='relu')(merged)#100
		output = Dense(y.shape[1], activation='softmax')(dense1)
		model  = Model([input1, input2], output)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model = model
		f = open("model.txt","w")
		f.write(str(X.shape[1])+","+str(X.shape[2])+"\n")
		f.write(str(y.shape[1]))
		f.close()

	def load_model_3(self, weights):
		f = open("model.txt","r")
		lines = f.readlines()
		X1,X2 = lines[0].split(",")
		y1    = lines[1]
		f.close()
		X1 = int(X1)
		X2 = int(X2)
		y1 = int(y1)

		input1 = Input(shape=(1,))
		input2 = Input(shape=(X1,X2))
		lstm1  = LSTM(1500)(input2)#512
		drop1  = Dropout(0.1)(lstm1)
		merged = Concatenate(axis=1)([input1, drop1])
		dense1 = Dense(100, activation='relu')(merged)#100
		output = Dense(y1, activation='softmax')(dense1)
		model  = Model([input1, input2], output)
		model.load_weights(weights)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model = model


	def set_model_4(self,X,y):
		input1 = Input(shape=(X.shape[1],X.shape[2]))
		lstm1  = LSTM(512, return_sequences=True)(input1)#512
		lstm2  = LSTM(128)(lstm1)
		output = Dense(y.shape[1], activation='softmax')(lstm2)
		model  = Model(inputs=input1, outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model = model
		f = open("model4.txt","w")
		f.write(str(X.shape[1])+","+str(X.shape[2])+"\n")
		f.write(str(y.shape[1]))
		f.close()

	def load_model_4(self, weights):
		f = open("model4.txt","r")
		lines = f.readlines()
		X1,X2 = lines[0].split(",")
		y1    = lines[1]
		f.close()
		X1 = int(X1)
		X2 = int(X2)
		y1 = int(y1)

		input1 = Input(shape=(X1,X2))
		lstm1  = LSTM(512, return_sequences=True)(input1)#512
		lstm2  = LSTM(128)(lstm1)
		output = Dense(y1, activation='softmax')(lstm2)
		model  = Model(inputs=input1, outputs=output)
		model.load_weights(weights)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model = model