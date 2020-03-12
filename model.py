import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
import torch.nn.functional as F
import os,pickle

class SharedCNNDiscriminator(nn.Module):
	
	def __init__(self, args, hdims):
		super(SharedCNNDiscriminator, self).__init__()
		self.device = None
		self.args = args
		self.discriminator0 = CNNDiscriminator(args, hdims)
		self.discriminator1 = CNNDiscriminator(args, hdims)
		self.dropout = nn.Dropout(args.dropout)
		self.linear = nn.Linear(self.args.n_filters*len(self.args.filter_sizes),1)
		
	def forward(self, disc, h_ori, h_tsf, ones, zeros):
		
		h_ori = torch.unsqueeze(h_ori,dim=1).permute(0,1,3,2)
		h_tsf = torch.unsqueeze(h_tsf,dim=1).permute(0,1,3,2)
		
		h_ori_feats = torch.Tensor([]).to(self.device)
		h_tsf_feats = torch.Tensor([]).to(self.device)
		
		if disc == '0':
			for k_sz, cell in zip(self.discriminator0.filter_sizes, self.discriminator0.conv_units):
				conv_ori = cell(h_ori)
				conv_tsf = cell(h_tsf)
				pooled_ori, _ = torch.max(torch.squeeze(conv_ori, dim=2)[:,:,k_sz-1:], dim=2)
				pooled_tsf, _ = torch.max(torch.squeeze(conv_tsf, dim=2)[:,:,k_sz-1:], dim=2)
				h_ori_feats = torch.cat([h_ori_feats,pooled_ori.view(h_ori.shape[0], -1)], dim = 1)
				h_tsf_feats = torch.cat([h_tsf_feats,pooled_tsf.view(h_tsf.shape[0], -1)], dim = 1)
		else:
			for k_sz, cell in zip(self.discriminator1.filter_sizes, self.discriminator1.conv_units):
				conv_ori = cell(h_ori)
				conv_tsf = cell(h_tsf)
				pooled_ori, _ = torch.max(torch.squeeze(conv_ori, dim=2)[:,:,k_sz-1:], dim=2)
				pooled_tsf, _ = torch.max(torch.squeeze(conv_tsf, dim=2)[:,:,k_sz-1:], dim=2)
				h_ori_feats = torch.cat([h_ori_feats,pooled_ori.view(h_ori.shape[0], -1)], dim = 1)
				h_tsf_feats = torch.cat([h_tsf_feats,pooled_tsf.view(h_tsf.shape[0], -1)], dim = 1)
		
		h_ori_feats = self.dropout(h_ori_feats)
		h_tsf_feats = self.dropout(h_tsf_feats)
			
		logits_ori = self.linear(h_ori_feats)
		logits_tsf = self.linear(h_tsf_feats)
		
		loss_d = F.binary_cross_entropy_with_logits(logits_ori,ones) + \
					F.binary_cross_entropy_with_logits(logits_tsf,zeros)

		loss_g = F.binary_cross_entropy_with_logits(logits_tsf,ones)
		
		return loss_d, loss_g
	
	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0]
		self.discriminator0 = self.discriminator0.to(*args,**kwargs)
		self.discriminator1 = self.discriminator1.to(*args,**kwargs)
		self.dropout = self.dropout.to(*args, **kwargs)
		self.linear = self.linear.to(*args, **kwargs)
		return self
	
class CNNDiscriminator(nn.Module):

	def __init__(self, args, hdims):
		super(CNNDiscriminator, self).__init__()
		self.device = None
		self.filter_sizes = args.filter_sizes
		self.n_filters = args.n_filters

		self.conv_units = nn.ModuleList([nn.Conv2d(
			in_channels=1, 
			out_channels=k_ft,
			kernel_size=[hdims, k_sz],
			stride=[hdims, 1],
			padding=[0,k_sz-1]
		) for k_sz, k_ft in zip(self.filter_sizes, [self.n_filters] * len(self.filter_sizes))])
	
	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0]
		self.conv_units = self.conv_units.to(*args, **kwargs)
		return self

class SharedLMDiscriminator(nn.Module):
	
	def __init__(self, args, vocab):
		super(SharedLMDiscriminator,self).__init__()
		self.device = None
		self.args = args
		self.hdims = args.hdims_disc
		self.num_layers = args.num_layers_disc
		self.dropout = nn.Dropout(args.dropout)
		self.project = nn.Linear(args.hdims_disc, vocab)
		self.discriminator0 = LMDiscriminator(args,vocab)
		self.discriminator1 = LMDiscriminator(args,vocab)
	
	def forward(self, disc, source_embed, target, input_seq, prob_seq, weights):
		batch_size = source_embed.shape[0]
		seq_len = source_embed.shape[1]
		
		hidden = self.init_hidden(batch_size)
		if disc == '0':
			output, hidden = self.discriminator0.lstm(source_embed, hidden)
		else:
			output, hidden = self.discriminator1.lstm(source_embed, hidden)
		
		logits = self.project(self.dropout(output))
		
		loss_d = F.cross_entropy(logits.view(-1,logits.shape[2]),target.view(-1), reduction='none')
		loss_d *= weights.view(-1)
		loss_d = torch.sum(loss_d) / (batch_size * seq_len)

		hidden_tsf = self.init_hidden(batch_size)
		if disc == '0':
			output_tsf, hidden_tsf = self.discriminator0.lstm(input_seq, hidden_tsf)
		else:
			output_tsf, hidden_tsf = self.discriminator1.lstm(input_seq, hidden_tsf)
		
		logits_tsf = self.project(self.dropout(output_tsf))
		
		logits_tsf = torch.softmax(logits_tsf,dim=2)
		logits_tsf = logits_tsf.view(-1, logits_tsf.shape[2])
		prob_seq = prob_seq.view(-1, prob_seq.shape[2])
		
		if self.args.generator_loss == 'normal':
			loss_g = -logits_tsf * torch.log(torch.clamp(prob_seq,min=1e-12))
		elif self.args.generator_loss == 'invert':
			loss_g = -prob_seq * torch.log(torch.clamp(logits_tsf,min=1e-12))
		elif self.args.generator_loss == 'kl':
			loss_g = prob_seq * torch.log(torch.clamp(prob_seq/torch.clamp(logits_tsf,min=1e-12),min=1e-12,max=1e12))
			
		loss_g = loss_g.sum(dim=1)
		loss_g = loss_g.mean()
		
		return loss_d, loss_g
	
# 	def pretrain(self, disc, source_embed, target):
# 		batch_size = source_embed.shape[0]
		
# 		hidden = self.init_hidden(batch_size)
# 		if disc == '0':
# 			output, hidden = self.discriminator0.lstm(source_embed, hidden)
# 		else:
# 			output, hidden = self.discriminator1.lstm(source_embed, hidden)
			
# 		logits = self.project(self.dropout(output))
# 		loss_lm = F.cross_entropy(logits.view(-1,logits.shape[2]),target.view(-1))
# 		return loss_lm
	
	def init_hidden(self, batch_size):
		return (torch.zeros(self.num_layers, batch_size, self.hdims).to(self.device),
					torch.zeros(self.num_layers, batch_size, self.hdims).to(self.device))
	
	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0]
		self.dropout = self.dropout.to(*args, **kwargs)
		self.project = self.project.to(*args, **kwargs)
		self.discriminator0 = self.discriminator0.to(*args,**kwargs)
		self.discriminator1 = self.discriminator1.to(*args,**kwargs)
		return self

class LMDiscriminator(nn.Module):

	def __init__(self, args, vocab):
		super(LMDiscriminator,self).__init__()
		self.device = None
		self.args = args
		self.hdims = args.hdims_disc
		self.num_layers = args.num_layers_disc
		self.lstm = nn.LSTM(
			input_size = args.word_dims,
			hidden_size = args.hdims_disc,
			num_layers = args.num_layers_disc,
			batch_first = True,
		)
		
	def to(self,*args,**kwargs):
		self = super().to(*args,**kwargs)
		self.device = args[0]
		self.lstm = self.lstm.to(*args, **kwargs)
		return self

class Model(nn.Module):

	def __init__(self, args, vocab, corpus):
		super(Model, self).__init__()
		self.device = None
		self.args = args
		self.gamma = args.gamma
		self.gamma_T = args.gamma
		self.gamma_min = args.gamma_min
		self.gamma_decay = args.gamma_decay
		
		self.max_len = args.max_len
		self.ydims = args.ydims
		self.zdims = args.zdims
		self.hdims = self.ydims + self.zdims
		self.num_layers = args.num_layers
		self.word_dims = args.word_dims
		self.beam_width = 50
		self.sos = corpus.word2id['<SOS>']
		self.unk = corpus.word2id['<UNK>']

		self.linear = nn.Linear(1,self.ydims)
		self.embed = nn.Embedding(vocab, self.word_dims)
		self.encoder = nn.LSTM(
				input_size = self.word_dims,
				hidden_size = self.hdims,
				num_layers = 1,
				batch_first = True,
			)
		self.generator = nn.LSTM(
				input_size = self.word_dims,
				hidden_size = self.hdims,
				num_layers = self.num_layers,
				batch_first = True,
			)
		self.dropout = nn.Dropout(args.dropout)
		self.project = nn.Linear(self.hdims, vocab)
# 		self.discriminators = SharedCNNDiscriminator(args,self.hdims)
		self.discriminators = SharedLMDiscriminator(args,vocab)
		
		# Attention
		self.v = nn.Linear(self.zdims, 1)
		self.W1 = nn.Linear(self.zdims, self.zdims)
		self.W2 = nn.Linear(self.zdims, self.zdims)
		self.attend = nn.Linear(2*self.zdims, self.zdims)
		
	def forward(self,data,decode=''):

		reverse_embed = self.embed(data['reverse'])
		source_embed = self.embed(data['source'])
		hidden = self.init_hidden(data['labels'])

		batch_size = data['source'].shape[0]
		seq_len = data['source'].shape[1]
		
		output1, hidden1 = self.encoder(reverse_embed,hidden)

		h_ori = self.init_hidden(data['labels'], hidden1)
		h_tsf = self.init_hidden(1 - data['labels'], hidden1)

		output2, _ = self.generator(source_embed, h_ori)
		
		# Attention
		if self.args.use_attention:
			output6 = self.multi_step_attention(output1, output2)
		else:
			output6 = output2
		
		logits_rec = self.project(self.dropout(output6))

		target = data['target']
		loss_rec = F.cross_entropy(logits_rec.view(-1,logits_rec.shape[-1]), target.view(-1), reduction='none')
		loss_rec *= data['weights'].view(-1)
		loss_rec = torch.sum(loss_rec) / (batch_size * seq_len)

		teach_h0 = h_ori[0]
		teach_h = torch.cat((teach_h0.view(teach_h0.shape[1],1,teach_h0.shape[2]),output2[:,:output6.shape[1]-1,:]),dim=1)

		soft_h_tsf, soft_logits_tsf, soft_input_tsf = self.rnn_decode(output1, h_tsf, source_embed[:,0,:], data['max_len'], softsample_word(self.embed.weight, self.gamma, self.device))

		half = batch_size // 2
		
		zeros, ones = data['labels'][half:], data['labels'][:half]
		
		loss_d0, loss_g0 = self.discriminators('0',source_embed[:half],data['target'][:half], soft_input_tsf[half:],soft_logits_tsf[half:],data['weights'][:half])
		loss_d1, loss_g1 = self.discriminators('1',source_embed[half:],data['target'][half:], soft_input_tsf[:half],soft_logits_tsf[:half],data['weights'][half:])
		
# 		loss_d0, loss_g0 = self.discriminators('0',teach_h[:half],soft_h_tsf[half:], ones, zeros)
# 		loss_d1, loss_g1 = self.discriminators('1',teach_h[half:],soft_h_tsf[:half], ones, zeros)
	
		if decode == 'greedy':
			seq_greedy = self.decode(output1, source_embed, h_tsf)
			seq_greedy_rec = self.decode(output1, source_embed, h_ori)
			return loss_rec, loss_d0, loss_g0, loss_d1, loss_g1, seq_greedy, seq_greedy_rec
		elif decode == 'beam':
			seq_greedy, seq_beam = self.decode(output1, source_embed, h_tsf, beam=True)
			seq_greedy_rec = self.decode(output1, source_embed, h_ori)
			return loss_rec, loss_d0, loss_g0, loss_d1, loss_g1, seq_greedy, seq_greedy_rec, seq_beam
		else:
			return loss_rec, loss_d0, loss_g0, loss_d1, loss_g1
	
	def multi_step_attention(self, encoded, output):
		
		encoded_content = encoded[:,:,self.ydims:]
		output_content = output[:,:,self.ydims:]
		
		output3 = torch.unsqueeze(encoded_content,dim=1).repeat(1,output_content.shape[1],1,1)
		output4 = torch.unsqueeze(output_content,dim=2).repeat(1,1,encoded_content.shape[1],1)
		matrix = torch.squeeze(self.v(self.W1(output3) + self.W2(output4)),dim=3) # batch_size * source_embed * reverse_embed
		attention_prob = torch.softmax(matrix,dim=2)
		context = torch.bmm(attention_prob, encoded_content) # batch_size * source_embed * hdims
		output5 = self.attend(torch.cat((context, output_content),dim=2))
		output6 = torch.cat((output5,output[:,:,:self.ydims]),dim=2)
		return output6
	
	def single_step_attention(self, encoded, output):
		
		encoded_content = encoded[:,:,self.ydims:]
		output_content = output[:,:,self.ydims:]
		
		output2 = output_content.repeat(1,encoded.shape[1],1)
		matrix = torch.squeeze(self.v(self.W1(encoded_content) + self.W2(output2)),dim=2)
		attention_prob = torch.softmax(matrix,dim = 1)
		context = torch.bmm(torch.unsqueeze(attention_prob,dim=1),encoded_content)
		output3 = self.attend(torch.cat((context, output_content),dim=2))
		output4 = torch.cat((output3,output[:,:,:self.ydims]),dim=2)
		return output4
	
	def rnn_decode(self, encoded, hidden, inp, max_len, func):
		h_seq, prob_seq = [], []
		inp = torch.unsqueeze(inp,dim=1)
		input_seq = []

		for i in range(max_len):
			h_seq.append(torch.unsqueeze(hidden[0][0],dim=1))
			input_seq.append(inp)
			output, hidden = self.generator(inp, hidden)
			
			# Attention
			if self.args.use_attention:
				output3 = self.single_step_attention(encoded, output)
			else:
				output3 = output
				
			inp, prob = func(self.project(self.dropout(output3)))
			prob_seq.append(prob)

		h_seq = torch.cat(h_seq,dim=1)
		prob_seq = torch.cat(prob_seq,dim=1)
		input_seq = torch.cat(input_seq,dim=1)

		return h_seq, prob_seq, input_seq
	
	def decode(self, encoded, source_embed, h_tsf, beam=False):
	
		batch_size = source_embed.shape[0]
		seq_len = source_embed.shape[1]
		
		# Greedy
		hidden_greedy = (h_tsf[0].clone(), h_tsf[1].clone())
		inp_embed = torch.unsqueeze(source_embed[:,0,:],dim=1)
		seq_greedy = []
		func = argmax_word(self.embed.weight,self.device)
		
		prob = [0]
		
		for i in range(seq_len):
			output_greedy, hidden_greedy = self.generator(inp_embed,hidden_greedy)
			
			if self.args.use_attention:
				output_greedy_attn = self.single_step_attention(encoded, output_greedy)
			else:
				output_greedy_attn = output_greedy
				
			inp_embed, inp_greedy , _ , p = func(self.project(output_greedy_attn))
			seq_greedy.append(torch.unsqueeze(inp_greedy,dim=1))
# 			print (inp_greedy.shape)
			prob.append(prob[-1] - torch.log(p)[0].item())
		
		seq_greedy = torch.cat(seq_greedy,dim=1).to(self.device)
# 		print (seq_greedy.shape)
		
		#Beam Search
		if beam:
			
			seq_beam = []
			for k in range(self.beam_width):
				single_beam = []
				inp = torch.unsqueeze(source_embed[:,0,:],dim=1)
				hidden_beam = h_tsf
				for j in range(seq_len):
					output_beam, hidden_beam = self.generator(inp, hidden_beam)
					if self.args.use_attention:
						output_beam_attn = self.single_step_attention(encoded, output_beam)
					else:
						output_beam_attn = output_beam
					_, _, softmax_beam, _ = func(self.project(output_beam_attn)/self.temp)
					topk_values, topk_indices = torch.topk(softmax_beam, self.topk, dim = 2)
					dist = torch.zeros_like(softmax_beam).scatter_(2, topk_indices, topk_values)
					sampled = torch.multinomial(dist.view(dist.shape[0],dist.shape[2]),1)
					single_beam.append(sampled.view(-1))
					inp = torch.index_select(self.embed.weight,0,sampled.view(-1)).view(batch_size,1,-1)

				single_beam = torch.stack(single_beam,dim=1)
				seq_beam.append(single_beam)
				
			seq_beam = torch.stack(seq_beam,dim=0)
			
			return seq_greedy, seq_beam
# 			beam_head = [[[self.sos for _ in range(batch_size)],torch.unsqueeze(source_embed[:,0,:],dim=1).clone(),h_tsf[0].clone(),h_tsf[1].clone(),[0 for _ in range(batch_size)],[[] for _ in range(batch_size)]] for _ in range(self.beam_width)]
# 			for i in range(seq_len):
# # 				print ('*'*50)
# 				list_of_best = [[] for _ in range(batch_size)]
# 				for cand in beam_head:
# 					output_beam, hidden_beam = self.generator(cand[1],(cand[2],cand[3]))
# 					if self.args.use_attention:
# 						output_beam_attn = self.single_step_attention(encoded, output_beam)
# 					else:
# 						output_beam_attn = output_beam
# 					_, _, softmax_beam, _ = func(self.project(output_beam_attn))
# 					topk_values, topk_indices = torch.topk(softmax_beam,self.beam_width,dim = 2)
					
# 					for j in range(batch_size):
# 						inp_embed = torch.index_select(self.embed.weight,0,topk_indices[j,0])
# 						curr_prob = cand[4][j]
# 						for k in range(self.beam_width):
# 							new_probs = curr_prob - torch.log(topk_values[j,0,k])
# 							list_of_best[j].append((hidden_beam[0][:,j].clone(),hidden_beam[1][:,j].clone(),topk_indices[j,0,k],torch.unsqueeze(inp_embed[k].clone(),dim=0),new_probs,cand[5][j] + [topk_indices[j,0,k].item()]))
# 				for j in range(len(list_of_best)):
# 					if i == 0:
# 						l = list_of_best[j]
# 					else:
# 						l = sorted(list_of_best[j],key=lambda x:x[4])
# 					for k in range(self.beam_width):
# 						beam_head[k][2][:,j] = l[k][0]
# 						beam_head[k][3][:,j] = l[k][1]
# 						beam_head[k][4][j] = l[k][4]
# 						beam_head[k][0][j] = l[k][2]
# 						beam_head[k][1][j] = l[k][3]
# 						beam_head[k][5][j] = l[k][5]
# 			for k in range(self.beam_width):
# 				for j in range(batch_size):
# 					beam_head[k][5][j] = torch.tensor(beam_head[k][5][j])
# 				beam_head[k][5] = torch.stack(beam_head[k][5])
# 			seq_beam = torch.stack([beam_head[x][5] for x in range(self.beam_width)]).to(self.device)
			
# 			return seq_greedy, seq_beam
		else:
			return seq_greedy

	def pretrain_lm(self, data):
		source_embed1 = self.embed(data['source0'])
		source_embed2 = self.embed(data['source1'])
		
		loss_lm0 = self.discriminators.pretrain('0',source_embed1,data['target0'])
		loss_lm1 = self.discriminators.pretrain('1',source_embed1,data['target1'])
		return loss_lm0, loss_lm1
	
	def init_hidden(self,labels,hidden=None):

		n = labels.shape[0]
		if hidden is None:
			hidden = (torch.cat((torch.unsqueeze(self.linear(labels),dim=0),torch.zeros(1,n,self.zdims).to(self.device)),dim=2),
						torch.cat((torch.unsqueeze(self.linear(labels),dim=0),torch.zeros(1,n,self.zdims).to(self.device)),dim=2))
		else:
			hidden = (torch.cat((torch.unsqueeze(self.linear(labels),dim=0),hidden[0][:,:,self.ydims:]),dim=2),
						torch.cat((torch.unsqueeze(self.linear(labels),dim=0),hidden[1][:,:,self.ydims:]),dim=2))
		return hidden
	
	def change_mode(self, mode):
		if mode == 'test':
			self.gamma_T = self.gamma
			self.gamma = 1.0
		else:
			self.gamma = self.gamma_T
	
	def change_temperature(self):
		self.gamma = max(self.gamma_min,self.gamma*self.gamma_decay)
		
	def load_embeddings(self, corpus):
		with open('data/embeddings.pkl','rb') as f:
			embeddings = pickle.load(f)
		count = 0
		for w in corpus.word2id:
			if w in embeddings:
				vec = torch.tensor([float(val) for val in embeddings[w]]).to(self.device)
				self.embed.weight.data[corpus.word2id[w]] = vec
				count += 1
		print ('Embeddings loaded for',count,'words')
		
	def update_unknown(self):
		# There are 6 artificial tokens
		mu = torch.mean(self.embed.weight.data[6:],dim=0)
		sigma = torch.std(self.embed.weight.data[6:],dim=0)
		distrib = torch.distributions.normal.Normal(mu, sigma)
		sample = distrib.sample()
		self.embed.weight.data[self.unk] = sample
		
	def load_temp(self,t,k):
		self.temp = t
		self.topk = k
	
	def to(self, *args, **kwargs):
		self = super().to(*args,**kwargs)
		self.device = args[0]
		self.linear = self.linear.to(*args,**kwargs)
		self.embed = self.embed.to(*args,**kwargs)
		self.encoder = self.encoder.to(*args,**kwargs)
		self.generator = self.generator.to(*args,**kwargs)
		self.project = self.project.to(*args,**kwargs)
		self.discriminators = self.discriminators.to(*args,**kwargs)
		self.v = self.v.to(*args,**kwargs)
		self.W1 = self.W1.to(*args,**kwargs)
		self.W2 = self.W2.to(*args,**kwargs)
		self.attend = self.attend.to(*args,**kwargs)
		return self
