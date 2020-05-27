import os
from collections import Counter
import pickle
import json
import random

random.seed(0)

SPECIAL_TOKENS = ['<SOS>','<EOS>','<PAD>','<UNK>','<NE>','<NUM>']

class Corpus():

	def __init__(self, args):
		self.args = args
		self.threshold = args.threshold
# 		if os.path.exists('data/data.pickle'):
# 			with open('data/data.pickle','rb') as f:
# 				[self.word2id, self.id2word, self.train0, self.train1, self.dev0, self.dev1, self.test0, self.test1. self.gen0, self.gen1] = pickle.load(f)
			
# 		else:
			
		self.word2id = {}
		self.id2word = []

		for tok in SPECIAL_TOKENS:
			self.add_token(tok)
			
		self.train0 = []
		self.train1 = []
		self.dev0 = []
		self.dev1 = []
		self.test0 = []
		self.test1 = []
		
		self.load_data()
			
# 			with open('data/data.pickle','wb') as f:
# 				pickle.dump([self.word2id, self.id2word, self.train0, self.train1, self.dev0, self.dev1, self.test0, self.test1, self.gen0, self.gen1],f)

	def load_data(self):
		with open('data/swbd/dev_0.txt',"r") as f:
			for line in f.readlines():
				self.dev0.append(self.add_line(ln))
		with open('data/swbd/dev_1.txt',"r") as f:
			for line in f.readlines():
				self.dev1.append(self.add_line(ln))
		with open('data/swbd/train_0.txt',"r") as f:
			for line in f.readlines():
				self.train0.append(self.add_line(ln))
		with open('data/swbd/train_1.txt',"r") as f:
			for line in f.readlines():
				self.train1.append(self.add_line(ln))
		with open('data/swbd/test_0.txt',"r") as f:
			for line in f.readlines():
				self.test0.append(self.add_line(ln))
		with open('data/swbd/test_1.txt',"r") as f:
			for line in f.readlines():
				self.test1.append(self.add_line(ln))
		#if self.args.treebank:
			#with open('data/treebank/all.txt','r') as f:
				#lines = f.readlines()

				#freq = Counter()
				#for ln in lines:
					#for w in ln.strip().split():
						#if w.isnumeric(): continue
						#if w.count('/NE/') > 0: continue
						#freq.update([w])
				#for w in freq:
					#if(freq[w] >= self.threshold):
						#self.add_token(w)

				#for ln in lines:
					#self.train0.append(self.add_line(ln))
		
		#if self.args.opus:
			#with open('data/OPUS/opus_dataset.txt','r') as f:
			#	lines = f.readlines()
#
#				freq = Counter()
#				for ln in lines:
#					for w in ln.strip().split():
#						if w.count('/NE/') > 0: continue
#						freq.update([w])
#				for w in freq:
#					if(freq[w] >= self.threshold):
#						self.add_token(w)
#
#				for ln in lines:
#					self.train0.append(self.add_line(ln))
#		
		#with open('data/moviecs/moviecs.json') as f:
			#data = json.load(f)
			
			#for i in range(len(data)):
				#if data[i]["dataset"] == "train":
					#self.train0.append(self.add_line(data[i]["mono"]))
					#if data[i]["gold"].strip() != "":
						#self.train1.append(self.add_line(data[i]["gold"]))
					#if self.args.mturk:
						#for j in range(len(data[i]["mturk"])):
							#self.train1.append(self.add_line(data[i]["mturk"][j]))
						
				#elif data[i]["dataset"] == "valid":
					#self.dev0.append(self.add_line(data[i]["mono"]))
					#self.dev1.append(self.add_line(data[i]["gold"]))
				
				#elif data[i]["dataset"] == "test":
					#self.test0.append(self.add_line(data[i]["mono"]))
					#self.test1.append(self.add_line(data[i]["gold"]))
	
	def add_line(self, l):
		for w in l.strip().split():
			self.add_token(w)
			
		return [self.word2id['<NUM>'] if w.isnumeric() else self.word2id['<NE>'] if w.count('/NE/') > 0 else self.word2id[w] if w in self.word2id else self.word2id['<UNK>'] for w in l.strip().split()]
	
	def add_token(self, tok):
		if not tok in self.word2id:
			self.word2id[tok] = len(self.id2word)
			self.id2word.append(tok)
			
	def load_data_for_lm(self, path):
		
		data = []
		
		assert (os.path.exists(path))
		
		with open(path,'r') as f:
			lines = f.readlines()
			for ln in lines:
				ln = ln.strip().split()
				for w in ln:
					data.append(self.word2id['<NUM>'] if w.isnumeric() else self.word2id['<NE>'] if w.count('/NE/') > 0 else self.word2id[w] if w in self.word2id else self.word2id['<UNK>'])
				data.append(self.word2id['<EOS>'])
				
		return data
	
	def print_data(self):
		
		for n,l in [('train0',self.train0),('train1',self.train1),('dev0',self.dev0),('dev1',self.dev1),('test0',self.test0),('test1',self.test1)]:
			with open('data/moviecs/' + n + '.txt','w') as f:
				for i in range(len(l)):
					f.write('%s\n'%(' '.join([self.id2word[w] for w in l[i]])))
					
class TransferCorpus():

	def __init__(self, corpus, path):
# 		self.args = args
# 		self.threshold = args.threshold
		self.word2id = corpus.word2id
		self.id2word = corpus.id2word
	
		self.data = []
		self.load_data(path)

	def load_data(self, path):
		with open(path,'r') as f:
			lines = [ln.strip() for ln in f.readlines()]
			for ln in lines:
				self.data.append(self.add_line(ln))
			
	def add_line(self, l):
		return [self.word2id['<NUM>'] if w.isnumeric() else self.word2id['<NE>'] if w.count('/NE/') > 0 else self.word2id[w] if w in self.word2id else self.word2id['<UNK>'] for w in l.strip().split()]
