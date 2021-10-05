class Model:
	def __init__(self):
		self.model = None

	def set_model_1(self):
		input1 = Input(shape=(1,))
		input2 = Input(shape=(X.shape[1],X.shape[2]))
		lstm1  = LSTM(256)(input2)
		merged = Concatenate(axis=1)([input1, lstm1])
		output = Dense(y.shape[1], activation='softmax')(merged)
		model  = Model(inputs=[input1, input2], outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model = model

	def set_model_2(self):
		input1 = Input(shape=(X_s2.shape[1],X_s2.shape[2]))
		input2 = Input(shape=(X.shape[1],X.shape[2]))
		merged = Add()([input1, input2])
		lstm1  = LSTM(512)(merged)
		output = Dense(y.shape[1], activation='softmax')(lstm1)
		model  = Model(inputs=[input1, input2], outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model = model
		
	def set_model_3(self):
		input1 = Input(shape=(1,))
		input2 = Input(shape=(X.shape[1],X.shape[2]))
		lstm1  = LSTM(1024)(input2)#512
		drop1  = Dropout(0.1)(lstm1)
		merged = Concatenate(axis=1)([input1, drop1])
		dense1 = Dense(100, activation='relu')(merged)#100
		output = Dense(y.shape[1], activation='softmax')(dense1)
		model  = Model(inputs=[input1, input2], outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model = model
		