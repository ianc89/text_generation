class Mapping:
	def __init__(self):
		self.chars_to_int = None
		self.int_to_chars = None
		self.n_vocab      = None
		self.seq_length   = None
		self.chars        = None

	def save(self):
		f = open("setting.txt","w")
		f.write(str(self.seq_length)+"\n")
		f.write(",".join(self.chars))
		f.close()

	def load(self):
		f = open("setting.txt","r")
		lines = f.readlines()
		self.create(lines[1].split(","),lines[0])
		f.close()

	def create(self, chars, seq_length):
		self.chars        = chars
		self.chars_to_int = dict((c, i) for i, c in enumerate(chars))
		self.int_to_chars = dict((i, c) for i, c in enumerate(chars))
		self.n_vocab      = len(chars)
		self.seq_length   = int(seq_length)

	def test(self):
		c = ["a","b","d","c"]
		s = 5
		self.create(c,s)
		self.save()
		self.load()