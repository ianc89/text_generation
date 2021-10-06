from settings import Mapping
from models import ModelClass
import numpy
import sys
filename = sys.argv[1]

m = Mapping()
m.load()

mo = ModelClass()
mo.load_model_3(filename)
model = mo.model

# This function allows us to manipulate the probabilities to add some creativity
def sample(preds, temperature):
	preds = preds.astype('float64')
	preds = preds / temperature
	exp_preds = numpy.exp(preds)
	preds = exp_preds / numpy.sum(exp_preds)
	#print (numpy.sum(preds))
	#print (preds)
	#probas = numpy.random.multinomial(1, preds.flatten(), size=1)
	probas = numpy.random.multinomial(100, preds.flatten(), size=1)
	#print (probas)
	#print (probas[0])
	return probas[0]

# Create a seed and cut at length of sequence
seed = "hello and welcome to the new world"
pattern = [m.chars_to_int[x] for x in seed]
pattern = pattern[:m.seq_length]

print ("Seed:")
print ("\"", ''.join([m.int_to_chars[value] for value in pattern]), "\"")
# generate characters
import sys
for i in range(100):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	#model 1
	# x_s = numpy.reshape([5], (1,1,1))
	#model 2
	# x_s = numpy.reshape([5,5,5,5,5], (1,len(pattern),1))
	#model 3
	x_s = numpy.reshape([4], (1,1,1))
	x = x / float(m.n_vocab)
	prediction = model.predict([x_s,x], verbose=0)
	index = numpy.argmax(prediction)
	result = m.int_to_chars[index]
	seq_in = [m.int_to_chars[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")

for smax in [0.01,0.02,0.05,0.1]:
	for i in range(500):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		#model 1
		# x_s = numpy.reshape([5], (1,1,1))
		#model 2
		# x_s = numpy.reshape([5,5,5,5,5], (1,len(pattern),1))
		#model 3
		x_s = numpy.reshape([3], (1,1,1))
		x = x / float(m.n_vocab)
		prediction = model.predict([x_s,x], verbose=0)
		index = numpy.argmax(sample(prediction,smax))
		result = m.int_to_chars[index]
		seq_in = [m.int_to_chars[value] for value in pattern]
		sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print ("\nDone.")