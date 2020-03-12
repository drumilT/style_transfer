import torch
import numpy as np
import random

def makeup(data, n):
	for i in range(n):
		data.append(data[i%len(data)])
	return data

def softsample_word(embed, gamma, device, eps=1e-20):
	
	def loop_func(logits):
		U = torch.empty(logits.shape).uniform_(0, 1).to(device)
		G = -torch.log(-torch.log(U + eps) + eps)
		softmax = torch.softmax((logits + G) / gamma, dim = 2)
		inp = torch.matmul(softmax, embed)
		return inp, softmax

	return loop_func

def softmax_word(embed, gamma, device):

	def loop_func(logits):
		softmax = torch.softmax(logits / gamma, dim = 2)
		inp = torch.matmul(softmax, embed)
		return inp, softmax

	return loop_func

def argmax_word(embed, device):

	def loop_func(logits):
		softmax = torch.softmax(logits, dim = 2)
		index = torch.argmax(softmax, dim = 2)
		prob = torch.max(softmax, dim = 2)
		index = torch.squeeze(index,dim=1)
		inp = torch.index_select(embed,0,index)
		inp = torch.unsqueeze(inp,dim=1)
		return inp, index, softmax, prob[0]

	return loop_func

class Accumulator(object):
    def __init__(self, div, names):
        self.div = div
        self.names = names
        self.n = len(self.names)
        self.values = [0.0] * self.n

    def clear(self):
        self.values = [0] * self.n

    def add(self, values):
        for i in range(self.n):
            self.values[i] += values[i] / self.div

    def output(self, s=''):
        if s:
            s += ' '
        for i in range(self.n):
            s += '%s %.2f' % (self.names[i], self.values[i])
            if i < self.n-1:
                s += ', '
        print (s)
		
def noise(x, unk, word_drop=0, k=3):
    n = len(x)
    for i in range(n):
        if random.random() < word_drop:
            x[i] = unk

    # slight shuffle such that |sigma[i]-i| <= k
    sigma = (np.arange(n) + (k+1) * np.random.rand(n)).argsort()
    return [x[sigma[i]] for i in range(n)]

def repackage_hidden(h):
		"""Wraps hidden states in new Tensors, to detach them from their history."""

		if isinstance(h, torch.Tensor):
			return h.detach()
		else:
			return tuple(repackage_hidden(v) for v in h)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
	
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits