import numpy as np
import argparse
import pprint
import torch
import time
import sys
import os
import math

from utils import *
from data import Corpus, TransferCorpus
from model import Model
argparser = argparse.ArgumentParser()

# argparser.add_argument('--data', type=str, default='data',help='data directory')
argparser.add_argument('--treebank', action='store_true',help='')
argparser.add_argument('--mturk', action='store_true',help='')
argparser.add_argument('--opus', action='store_true',help='')
argparser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
argparser.add_argument('--mode', choices=['pretrain','train','test','generate'], default='train', help='')
argparser.add_argument('--log_interval', type=int, default=100, help='')

argparser.add_argument('--ydims', type=int, default=100, help='')
argparser.add_argument('--zdims', type=int, default=300, help='')
argparser.add_argument('--word_dims', type=int, default=300, help='')
argparser.add_argument('--num_layers', type=int, default=1, help='')
argparser.add_argument('--batch_size', type=int, default=16, help='')
argparser.add_argument('--epochs', type=int, default=100, help='')
argparser.add_argument('--dropout', type=float, default=0.2, help='')
argparser.add_argument('--lr_decay', type=float, default=0.8, help='')
argparser.add_argument('--gamma', type=float, default=1.0, help='')
argparser.add_argument('--gamma_decay', type=float, default=0.5, help='')
argparser.add_argument('--gamma_min', type=float, default=1.0, help='') # ?

argparser.add_argument('--grad_clip', type=float, default=5.0, help='')
argparser.add_argument('--rho', type=float, default=0.1, help='')
argparser.add_argument('--max_len', type=int, default=20, help='') #?
argparser.add_argument('--lr1', type=float, default=1.0, help='')
argparser.add_argument('--lr2', type=float, default=1.0, help='')
argparser.add_argument('--optim1', type=str, default='sgd', help='')
argparser.add_argument('--optim2', type=str, default='sgd', help='')
argparser.add_argument('--delta1', type=float, default=2.5, help='')
argparser.add_argument('--delta2', type=float, default=2.5, help='')
argparser.add_argument('--disable', action='store_true', help='')
argparser.add_argument('--use_attention', action='store_true', help='')
argparser.add_argument('--pretrained_embeddings', action='store_true', help='')
# CNN Discriminator
argparser.add_argument('--filter_sizes', type=lambda s: [int(item) for item in s.split(',')], default=[1,2,3,4,5], help='comma separated list of filter sizes/widths')
argparser.add_argument('--n_filters', type=int, default=128, help='comma separated list of filter features')

argparser.add_argument('--num_layers_disc', type=int, default=1, help='')
argparser.add_argument('--hdims_disc', type=int, default=500, help='')
argparser.add_argument('--generator_loss', choices=['kl','normal','invert'], default='normal', help='')
argparser.add_argument('--iters', type=int, default=100, help='')
argparser.add_argument('--lr_lm', type=float, default=0.01, help='')
argparser.add_argument('--seq_len_lm', type=int, default=20, help='')
argparser.add_argument('--batch_size_lm', type=int, default=100, help='')
argparser.add_argument('--optim_lm', type=str, default='sgd', help='')
argparser.add_argument('--threshold', type=int, default=0, help='')
argparser.add_argument('--cuda', action='store_true', help='')
argparser.add_argument('--save', type=str, default='saved/', help='')
argparser.add_argument('--load', type=str, default='', help='')
argparser.add_argument('--load_lm', type=str, default='', help='')
argparser.add_argument('--samples', action='store_true', help='')
argparser.add_argument('--temp', type=float, default=10.0, help='')
argparser.add_argument('--topk', type=int, default=2, help='')

args = argparser.parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)

if not os.path.exists(args.save):
	os.makedirs(args.save)
	
# class Logger(object):
# 	def __init__(self):
# 		self.terminal = sys.stdout
# 		self.log = open(os.path.join(args.save,"log.txt"), "w")

# 	def write(self, message):
# 		self.terminal.write(message)
# 		self.log.write(message)  
	
# 	def flush(self):
# 		self.log.flush()

# sys.stdout = Logger()

if torch.cuda.is_available():
	if not args.cuda:
		print ("CUDA is available, run with --cuda")
		args.cuda = True
	else:
		torch.cuda.manual_seed(args.seed)
		
print ('-'*50)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))
print ('-'*50)

corpus = Corpus(args)
lmmrl_corpus = TransferCorpus(corpus,'data/outdomain/lmmrl.txt')
treebank_corpus = TransferCorpus(corpus,'data/outdomain/treebank.txt')
opus_corpus = TransferCorpus(corpus,'data/outdomain/opus.txt')

print (len(corpus.id2word))

DEVICE = torch.device('cuda' if args.cuda else 'cpu')

print ('| Train | %d | %d |'%(len(corpus.train0),len(corpus.train1)))
print ('| Dev   | %d | %d |'%(len(corpus.dev0),len(corpus.dev1)))
print ('| Test  | %d | %d |'%(len(corpus.test0),len(corpus.test1)))

def get_batch(data, labels, min_length = 5):

	max_len = max(min_length,max([len(sent) for sent in data]))
	reverse, source, target, weights = [], [], [], []

	sos = corpus.word2id['<SOS>']
	eos = corpus.word2id['<EOS>']
	pad = corpus.word2id['<PAD>']

	for sent in data:
		padding = [pad] * (max_len - len(sent))
		_sent = noise(sent, corpus.word2id['<UNK>'])
		if args.use_attention:
			reverse.append(_sent + padding)
		else:
			reverse.append(padding + _sent[::-1])
		source.append([sos] + sent + padding)
		target.append(sent + [eos] + padding)
		weights.append([1.0] * (len(sent) + 1) + [0.0] * (max_len - len(sent)))

	reverse = torch.tensor(reverse,dtype=torch.long)
	source = torch.tensor(source,dtype=torch.long)
	target = torch.tensor(target,dtype=torch.long)
	labels = torch.tensor([labels],dtype=torch.float).permute(1,0)
	weights = torch.tensor(weights,dtype=torch.float)

	return {'reverse':reverse, 'source':source, 'target':target, 'weights':weights, 'labels':labels, 'max_len':max_len + 1, 'len':len(data)}

def batchify(data1, data2, batch_size):

	if len(data1) < len(data2):
		data1 = makeup(data1, len(data2) - len(data1))
	elif len(data2) < len(data1):
		data2 = makeup(data2, len(data1) - len(data2))

	data1 = sorted(data1, key=lambda x : len(x))
	data2 = sorted(data2, key=lambda x : len(x))

	n = len(data1) // batch_size
	data = []

	for i in range(n):
		s = i*batch_size
		t = (i+1)*batch_size
		data.append(get_batch(data1[s:t] + data2[s:t],[0] * (t-s) + [1] * (t-s)))
	
	random.shuffle(data)
	return data

def batchify_mono(data1, batch_size):
	data1 = sorted(data1, key=lambda x : len(x))
	n = len(data1) // batch_size
	data = []

	for i in range(n):
		s = i*batch_size
		t = (i+1)*batch_size
		data.append(get_batch(data1[s:t],[0] * (t-s)))
	
	random.shuffle(data)
	return data

def batchify_lm(data0, data1, batch_size, seq_len):
	n = min(len(data0),len(data1)) // (batch_size * seq_len)
	source0 = data0[:n * batch_size * seq_len]
	target0 = data0[1:n * batch_size * seq_len + 1]
	source1 = data1[:n * batch_size * seq_len]
	target1 = data1[1:n * batch_size * seq_len + 1]
	
	assert len(source0) == len(target0)
	assert len(source1) == len(target1)
	
	source0 = torch.tensor(source0)
	target0 = torch.tensor(target0)
	source1 = torch.tensor(source1)
	target1 = torch.tensor(target1)
	
	source0 = source0.reshape(-1,batch_size,seq_len)
	target0 = target0.reshape(-1,batch_size,seq_len)
	
	source1 = source1.reshape(-1,batch_size,seq_len)
	target1 = target1.reshape(-1,batch_size,seq_len)
	
	dataset = []
	for i in range(n):
		dataset.append({'source0':source0[i],'target0':target0[i],'source1':source1[i],'target1':target1[i]})
	return dataset

def train_lm(model, data):
	total_loss_lm0 = 0.0
	total_loss_lm1 = 0.0
	t1 = time.time()
	for j,batch in enumerate(data):
		loss_lm0, loss_lm1 = model.pretrain_lm(batch)
			
		total_loss_lm0 += loss_lm0.item()
		total_loss_lm1 += loss_lm1.item()
			
		loss = loss_lm0 + loss_lm1
	
		optimiser_lm.zero_grad()
		loss.backward()
		optimiser_lm.step()
			
		if (j + 1) % args.log_interval == 0:
			total_loss_lm0 /= args.log_interval
			total_loss_lm1 /= args.log_interval
			print ('| pretrain | batch {:d}/{:d} | time {:f} | loss_total {:6.4f} | loss_lm0 {:6.4f} | loss_lm1 {:6.4f} |'.format(j + 1,len(data),(time.time() - t1),(total_loss_lm0 + total_loss_lm1),total_loss_lm0, total_loss_lm1, ))
			t1 = time.time()
			total_loss_lm0 = 0
			total_loss_lm1 = 0

def test_lm(model, data):
	
	with torch.no_grad():
		val_loss_lm0 = 0
		val_loss_lm1 = 0
		for j,batch in enumerate(data):
			loss_lm0, loss_lm1 = model.pretrain_lm(batch)
			val_loss_lm0 += loss_lm0.item() 
			val_loss_lm1 += loss_lm1.item()
				
		val_loss_lm0 /= len(data)
		val_loss_lm1 /= len(data)
		
		return val_loss_lm0, val_loss_lm1
	
train_data = batchify(corpus.train0, corpus.train1, args.batch_size)
combined_data = batchify(corpus.train0 + corpus.dev0, corpus.train1 + corpus.dev1, args.batch_size)
lmmrl_data = batchify_mono(lmmrl_corpus.data, args.batch_size)
treebank_data = batchify_mono(treebank_corpus.data, args.batch_size)
print ('| batch | combined %d |'%(len(combined_data)))


if args.mode == 'pretrain':
	train_data_lm = batchify_lm(corpus.lm_train0, corpus.lm_train1, args.batch_size_lm, args.seq_len_lm)
	dev_data_lm = batchify_lm(corpus.lm_dev0, corpus.lm_dev1, args.batch_size_lm, args.seq_len_lm)

	print ('| pretrain | train %d | dev %d |'%(len(train_data_lm),len(dev_data_lm)))

model = Model(args, len(corpus.id2word), corpus).to(DEVICE)

if args.pretrained_embeddings:
	model.load_embeddings(corpus)

if args.load != '':
	model = torch.load(args.load)
	model = model.to(DEVICE)
	model.load_temp(args.temp,args.topk)
	print ('Model Loaded')
	
elif args.load_lm != '':
	model_dict = model.state_dict()
	pretrained_dict = torch.load(args.load_lm)
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	print ('LM Loaded')

# print model summary
print("")
print("="*70)
print("{:>30} {:>20} {:>15}".format("Name", "Shape", "#Params"))
print("="*70)
total_params = 0
for n, p in model.named_parameters():
	if p.requires_grad:
		print("{:>30} {:>20} {:>15}".format(n, str(list(p.shape)), str(p.numel())))
		total_params += p.numel()
print("="*70)
print("Total parameters: %d (%.2f M)" % (total_params, total_params / 1e6))
print("")

# if args.mode == 'pretrain':
# 	temp_loss1, temp_loss2 = test_lm(model,dev_data_lm)
# 	print ('Initial LM loss/ppl | val_loss0 {:6.4f} | val_ppl0 {:6.2f} | val_loss1 {:6.4f} |  val_ppl1 {:6.2f} |'.format(temp_loss1,math.exp(temp_loss1),temp_loss2,math.exp(temp_loss2)))

model_params = list(model.linear.parameters()) + list(model.embed.parameters()) + \
			list(model.encoder.parameters()) + list(model.generator.parameters()) + \
			list(model.project.parameters()) + list(model.v.parameters()) + \
			list(model.W1.parameters()) + list(model.W2.parameters()) + list(model.attend.parameters())

discriminator_params = list(model.discriminators.parameters())

if args.optim1 == 'sgd':
	optimiser1 = torch.optim.SGD(model_params, lr = args.lr1)
else:
	optimiser1 = torch.optim.Adam(model_params, lr = args.lr1)
if args.optim2 == 'sgd':
	optimiser2 = torch.optim.SGD(discriminator_params, lr = args.lr2)
else:
	optimiser2 = torch.optim.Adam(discriminator_params, lr = args.lr2)
	
if args.optim_lm == 'sgd':
	optimiser_lm = torch.optim.SGD(model.parameters(), lr = args.lr_lm)
else:
	optimiser_lm = torch.optim.Adam(model.parameters(), lr = args.lr_lm)

# scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser1,factor=args.lr_decay,patience=5,min_lr=1e-4)
# scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser2,factor=args.lr_decay,patience=5,min_lr=1e-4)

optimiser_gen = torch.optim.SGD(model.parameters(), lr = 1.0)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimiser1,gamma=args.lr_decay,step_size=10)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimiser2,gamma=args.lr_decay,step_size=10)

scheduler_gen = torch.optim.lr_scheduler.StepLR(optimiser_gen,gamma=0.8,step_size=50)

def data_to_gpu(batch):
	for k in batch.keys():
		if type(batch[k]) == torch.Tensor:
			batch[k] = batch[k].to(DEVICE)
	return batch

def test(model, data, samples='', file=None):
	
	with torch.no_grad():
		losses = [0 for _ in range(5)]
	
		# Writing samples to file
		if samples != '':
			f1 = open(file,'w')
		
		for i,batch in enumerate(data):
			batch = data_to_gpu(batch)
			if samples == 'greedy':
				loss_rec, loss_d0, loss_g0, loss_d1, loss_g1, seq_greedy, seq_greedy_rec = model.forward(batch, decode='greedy')
				batch_size = batch['source'].shape[0]
				for j in range(batch_size):
					if batch['labels'][j] == 0:
						source = [corpus.id2word[x] for x in batch['source'][j]]
						greedy = [corpus.id2word[x] for x in seq_greedy[j]]
						greedy_rec = [corpus.id2word[x] for x in seq_greedy_rec[j]]
						f1.write('original - %s\n'%(' '.join(source)))
						f1.write('transfer - %s\n'%(' '.join(greedy)))
						f1.write('reconstruction - %s\n\n'%(' '.join(greedy_rec)))
				
			elif samples == 'beam':
				loss_rec, loss_d0, loss_g0, loss_d1, loss_g1, seq_greedy, seq_greedy_rec, seq_beam = model.forward(batch, decode='beam')
				batch_size = batch['source'].shape[0]
				for j in range(batch_size):
					if batch['labels'][j] == 0:
						source = [corpus.id2word[x] for x in batch['source'][j]]
						greedy = [corpus.id2word[x] for x in seq_greedy[j]]
						beam = [[corpus.id2word[x] for x in seq_beam[k,j]] for k in range(10)]
						greedy_rec = [corpus.id2word[x] for x in seq_greedy_rec[j]]
						f1.write('original - %s\n'%(' '.join(source)))
						f1.write('transfer - %s\n'%(' '.join(greedy)))
						for k in range(10):
							f1.write('beam - %s\n'%(' '.join(beam[k])))
						f1.write('reconstruction - %s\n\n'%(' '.join(greedy_rec)))
			else:
				loss_rec, loss_d0, loss_g0, loss_d1, loss_g1 = model.forward(batch)
			
			loss_disc = loss_d0	+ loss_d1
			loss_adv = loss_g0 + loss_g1
			loss_total = loss_rec + args.rho * loss_adv
			losses = [sum(x) for x in zip(losses,[loss_total.item(), loss_rec.item(), loss_adv.item(), loss_d0.item(), loss_d1.item()])]
			
			torch.cuda.empty_cache()
				
		
		if samples != '':
			f1.close()
				
		losses = [x / len(data) for x in losses]
	
		return losses
	
def train(model, data, train_adversarial=False):
	
	total_losses = [0 for _ in range(5)]
	losses = [0 for _ in range(5)]
	step = 0
	t1 = time.time()
	value = {}
	grad_adv = {}
	grad_rec = {}
		
	for i,batch in enumerate(data):
		batch = data_to_gpu(batch)
		loss_rec, loss_d0, loss_g0, loss_d1, loss_g1 = model.forward(batch)
		loss_disc = loss_d0	+ loss_d1
		loss_adv = loss_g0 + loss_g1
		loss_total = loss_rec + args.rho * loss_adv

		losses = [sum(x) for x in zip(losses,[loss_total.item(), loss_rec.item(), loss_adv.item(), loss_d0.item(), loss_d1.item()])]
		total_losses = [sum(x) for x in zip(total_losses,[loss_total.item(), loss_rec.item(), loss_adv.item(), loss_d0.item(), loss_d1.item()])]
		
		optimiser2.zero_grad()
		loss_disc.backward(retain_graph=True)
		optimiser2.step()

# 		for name, params in model.named_parameters():
# 			value[name] = value.get(name,0) + torch.mean(torch.abs(params.data))
# 		optimiser1.zero_grad()
# 		loss_adv.backward(retain_graph=True)
# 		for name, params in model.named_parameters():
# 			grad_adv[name] = grad_adv.get(name,0) + torch.mean(torch.abs(params.grad))
# 		optimiser1.zero_grad()
# 		loss_rec.backward(retain_graph=True)
# 		for name, params in model.named_parameters():
# 			grad_rec[name] = grad_rec.get(name,0) + torch.mean(torch.abs(params.grad))
			
		optimiser1.zero_grad()
		
		if train_adversarial and not args.disable:
			loss_total.backward()
		else:
			loss_rec.backward()
			
		torch.nn.utils.clip_grad_norm_(model_params,args.grad_clip)
		
		optimiser1.step()

		step += 1
		torch.cuda.empty_cache()

		if step % args.log_interval == 0:
			print3(step, len(data), time.time() - t1, losses)
			losses = [0 for _ in range(5)]

# 			for k in value.keys():
# 				print (k,value[k]/args.log_interval,grad_rec[k]/args.log_interval,grad_adv[k]/args.log_interval)
# 				value[k] = 0
# 				grad_rec[k] = 0
# 				grad_adv[k] = 0
			
			t1 = time.time()
	
	return [x/step for x in total_losses]

def print1(curr_epoch, total_epoch, val_loss_total, val_loss1, val_loss2):
	print ('| val epoch {:d}/{:d} | lr {:6.4f} | total {:6.4f} | lm0 {:6.4f} | ppl0 {:6.4f} | lm1 {:6.4f} | ppl1 {:6.4f} |'.format(curr_epoch, total_epoch, get_lr(optimiser_lm), val_loss_total, val_loss1, math.exp(val_loss1), val_loss2, math.exp(val_loss2)))

def print2(curr_epoch, total_epoch, val_loss,s="val_loss"):
	print ('{} {:d}/{:d} | total {:6.4f} | rec {:6.4f} | adv {:6.4f} | d0 {:4.3f} | d1 {:4.3f} | lr0 {:4.4f} | lr1 {:4.4f} '.format(s,curr_epoch, total_epoch, val_loss[0],val_loss[1],val_loss[2],val_loss[3],val_loss[4],get_lr(optimiser1), get_lr(optimiser2)))

def print3(step,total,t,losses):
	print('| batch {:d}/{:d} | time {:f} | loss {:6.4f} | rec {:6.4f} | adv {:6.4f} | d0 {:4.3f} | d1 {:4.3f} |'.format(step,total,t,losses[0]/args.log_interval,losses[1]/args.log_interval,losses[2]/args.log_interval,losses[3]/args.log_interval,losses[4]/args.log_interval))
	
def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']
			
def main():
	
	if args.mode == 'pretrain':
		best_val_loss = 1000
		for epoch in range(1,args.iters+1):
			train_lm(model, train_data_lm)
			val_loss1, val_loss2 = test_lm(model, dev_data_lm)
			val_loss_total = val_loss1 + val_loss2
			
			scheduler_lm.step()
			print1(epoch, args.iters, val_loss_total, val_loss1, val_loss2)
			
			if val_loss_total < best_val_loss:
				pretrained_dict = {k:v for k,v in model.state_dict().items() if k.count('embed') > 0 or k.count('discriminator') > 0}
				torch.save(pretrained_dict, os.path.join(args.save,'pretrained.pt'))
				best_val_loss = val_loss_total
			
		print ('best val loss - {:6.4f}'.format(best_val_loss))
				
	elif args.mode == 'train':
		best_val_loss = 1000
		train_adversarial = False
		for epoch in range(1,args.epochs+1):

			model.change_mode('train')
			train_loss = train(model,combined_data, train_adversarial)
			
			if train_loss[3] < args.delta1 and train_loss[4] < args.delta2:
				train_adversarial = True
			else:
				train_adversarial = False
			
			model.change_mode('test')
			
			# Validation
			if epoch % 10 == 0 and args.samples:
				# Need to check whether fits in memory or not
				best_model = torch.load(os.path.join(args.save,'model.pt'))
				best_model = best_model.to(DEVICE)
				combined_loss = test(best_model,combined_data,samples='greedy',file=os.path.join(args.save,"trainsamples-" + str(epoch) + '.txt'))
				print2(epoch, args.epochs, combined_loss, "best_val_loss")
				del best_model
				
			else:
				combined_loss = test(model,combined_data)	
				
			print2(epoch, args.epochs, combined_loss)

# 			if combined_loss[1] < best_val_loss:
# 				best_val_loss = combined_loss[1]
			torch.save(model,os.path.join(args.save,'model-' + str(epoch) + '.pt'))
			
			model.change_temperature()
			model.update_unknown()
			
			scheduler1.step()
			scheduler2.step()
		
		best_model = torch.load(os.path.join(args.save,'model.pt'))
		best_model = best_model.to(DEVICE)
		combined_loss = test(best_model, combined_data)
		print2(args.epochs + 1, args.epochs, combined_loss, "best_val_loss")
		
	elif args.mode == 'test':
		print ("TESTING")
		combined_loss = test(model,combined_data,samples='greedy',file=os.path.join(args.save,"trainsamples.txt"))
		print2(args.epochs + 1, args.epochs, combined_loss, "best_val_loss")

	else:
		test(model,train_data,samples='beam',file=os.path.join(args.save,"train.txt"))
# 		test(model,lmmrl_data,samples='beam',file=os.path.join(args.save,"lmmrl.txt"))
# 		test(model,treebank_data,samples='beam',file=os.path.join(args.save,"treebank.txt"))
		
main()