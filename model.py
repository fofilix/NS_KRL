import math
import random

import numpy as np
from info_nce import InfoNCE
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from ordered_set import OrderedSet
from functools import partial
from helper import *
from dpw import depthwise_separable_conv
from tqdm import tqdm
import pysnooper
from datetime import datetime


def get_samples_idxes(sub, pred, label, obj, device):
	row_range = torch.arange(sub.shape[0], device=device)
	target_pred = pred[row_range, obj]  
	pred = torch.where(label.byte(), torch.zeros_like(pred), pred)  
	pred[row_range, obj] = target_pred 
	samples_idxes = torch.argsort(pred, dim=1, descending=True)

	return samples_idxes


# def func(x, sub, rel, sr2o_all, n_neg):
# 	neg_obj_temp = []
#
# 	item, max_neg_num = x[1], 100     # feel free to set max_neg_num
#
# 	assert n_neg <= max_neg_num
# 	# For DEBUG
#
# 	result = OrderedSet(item.cpu().numpy()[:max_neg_num]) - OrderedSet(sr2o_all[(sub[x[0]].item(), rel[x[0]].item())])
# 	neg_obj_temp = list(result)[:n_neg]
#
# 	return neg_obj_temp


class BinaryCrossEntropyLoss(torch.nn.Module):
	"""This class implements :class:`torch.nn.Module` interface."""

	def __init__(self, p):
		super().__init__()
		self.p = p
		self.sig = torch.nn.Sigmoid()
		self.loss = torch.nn.BCELoss(reduction='mean')

	def forward(self, positive_triplets, negative_triplets):
		"""
        Parameters
        ----------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the true triplets as returned by the `forward` methods
            of the models.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the negative triplets as returned by the `forward`
            methods of the models.
        Returns
        -------
        loss: torch.Tensor, shape: (n_facts, dim), dtype: torch.float
            Loss of the form :math:`-\\eta \\cdot \\log(f(h,r,t)) +
            (1-\\eta) \\cdot \\log(1 - f(h,r,t))` where :math:`f(h,r,t)`
            is the score of the fact and :math:`\\eta` is either 1 or
            0 if the fact is true or false.
        """
		if self.p.lbl_smooth != 0.0:
			return self.loss(self.sig(positive_triplets),
						 (1-self.p.lbl_smooth)*torch.ones_like(positive_triplets) + self.p.lbl_smooth/self.p.num_rel) + \
				   self.loss(self.sig(negative_triplets),
						 torch.zeros_like(negative_triplets) + self.p.lbl_smooth/self.p.num_rel)
		else:
			return self.loss(self.sig(positive_triplets),
							 torch.ones_like(positive_triplets)) + \
				   self.loss(self.sig(negative_triplets),
							 torch.zeros_like(negative_triplets))



class SeparableConv2d(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size:tuple, stride, padding):
		super(SeparableConv2d, self).__init__()
		self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
		self.pointwise = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=1, padding=0)

	def forward(self, x):
		return self.pointwise(self.conv1(x))


class Way1(torch.nn.Module):
	"""changed convkb"""
	def __init__(self, params, chequer_perm_3vec):
		super(Way1, self).__init__()
		self.params = params
		self.chequer_perm_3vec = chequer_perm_3vec

		# -*-*-*- convolution -*-*-*-
		# self.way1_cnn = torch.nn.Sequential(
		# 	torch.nn.Conv2d(1, 32, kernel_size=(3,5), stride=1, padding=0),    
		# 	torch.nn.BatchNorm2d(32),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Conv2d(32, 32, kernel_size=(3,5), stride=1, padding=0),   
		# 	torch.nn.BatchNorm2d(32),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout(0.2),
		# 	torch.nn.Conv2d(32, 64, kernel_size=(3,5), stride=1, padding=0),  
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout2d(0.2),
		# 	torch.nn.Conv2d(64, 64, kernel_size=(3,5), stride=1, padding=0),  
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout(0.2),
		# )

		# -*-*-*- Depthwise separable convolution -*-*-*-
		self.way1_cnn = torch.nn.Sequential(
			SeparableConv2d(1, 32, kernel_size=(3, 5), stride=1, padding=0),  
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			SeparableConv2d(32, 32, kernel_size=(3, 5), stride=1, padding=0),  
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
			SeparableConv2d(32, 64, kernel_size=(3, 5), stride=1, padding=0),  
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(0.2),
			SeparableConv2d(64, 64, kernel_size=(3, 5), stride=1, padding=0),  
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
		)


		self.way1_fc = torch.nn.Linear(64*12*14, 1)    

	def forward(self, batchsize, sub_emb, rel_emb, obj_emb):
		"""

		Parameters
		----------
		batchsize:
		sub_emb: 
		rel_emb: 
		obj_emb:

		Returns
		-------

		"""
		comb_emb_hrt = torch.cat([sub_emb, rel_emb, obj_emb], dim=1)  
		chequer_perm_hrt = comb_emb_hrt[:, self.chequer_perm_3vec]  
		integrate_inp = chequer_perm_hrt.reshape(batchsize, self.params.perm_3vec, self.params.k_h, -1)   


		x = self.way1_cnn(integrate_inp)   
		x = x.flatten(1)                   
		x = self.way1_fc(x)                
		x = F.sigmoid(x)

		return x




class BPRLoss(torch.nn.Module):
	"""
	"""

	def __init__(self, reduction='mean'):
		super(BPRLoss, self).__init__()
		self.reduction = reduction
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, pos_preds, neg_preds):
		distance = pos_preds - neg_preds
		loss = torch.sum(torch.log(1e-6+(1 + torch.exp(-distance))))  
		if self.reduction == 'mean':
			loss = loss.mean()

		return loss


def dynamic_weighted_binary_crossentropy_withlogits(l, y_pred, y_true, alpha=0.5):
	def loss(y_pred, y_true):
		w_neg = torch.sum(y_true).item() / l
		w_pos = 1 - w_neg
		r = 2 * w_neg * w_pos
		w_neg /= r
		w_pos /= r

		b_ce = F.binary_cross_entropy_with_logits(y_pred, y_true)
		w_b_ce = b_ce * y_true * w_pos + b_ce * (1 - y_true) * w_neg
		return torch.mean(w_b_ce) * alpha + torch.mean(b_ce) * (1 - alpha)

	return loss(y_pred, y_true)


class KGML(nn.Module):
	def __init__(self, params):
		super(KGML, self).__init__()
		self.params = params
		# self.k_s = (5, 5)
		# self.KGML_cnn = nn.Sequential(
		# 	nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.k_s, padding=0),
		# 	nn.BatchNorm2d(32),
		# 	nn.ReLU(),
		# 	nn.Dropout(0.2),
		# 	SeparableConv2d(in_channels=32, out_channels=64, kernel_size=self.k_s, padding=0, stride=1),
		# 	nn.BatchNorm2d(64),
		# 	nn.ReLU(),
		# 	nn.Dropout(0.1),
		# 	SeparableConv2d(in_channels=64, out_channels=128, kernel_size=self.k_s, padding=0, stride=1),
		# 	nn.BatchNorm2d(128),
		# 	nn.ReLU(),
		# )
		# filtered_shape = (18, 8)
		# self.KGML_fc = nn.Linear(128*filtered_shape[0]*filtered_shape[1], self.params.num_rel)

		self.KGML_ffn = nn.Sequential(
			nn.Linear(self.params.embed_dim*2, self.params.embed_dim*3),
			nn.Dropout(0.3),
			nn.Linear(self.params.embed_dim*3, int(self.params.embed_dim*1.5)),
			nn.BatchNorm1d(int(self.params.embed_dim*1.5)),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(int(self.params.embed_dim * 1.5), int(self.params.embed_dim * 0.75)),
			nn.BatchNorm1d(int(self.params.embed_dim * 0.75)),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(int(self.params.embed_dim * 0.75),self.params.num_rel),
			nn.Sigmoid(),
		)


	def forward(self, sub_emb, obj_emb, label):
		# x = torch.cat([sub_emb, sub_emb*obj_emb, obj_emb], dim=1)
		# x = x.view(-1, 1, 30, 20)
		# x = self.KGML_cnn(x)
		# x = x.flatten(1)
		# x = self.KGML_fc(x)

		x = torch.cat([sub_emb, obj_emb], dim=1)
		x = self.KGML_ffn(x)

		loss = dynamic_weighted_binary_crossentropy_withlogits(self.params.num_rel, x, label)

		# loss = InfoNCE(negative_mode='paired')
		# batch_size, num_negative, embedding_size = 32, 6, 128
		#
		# query = torch.randn(batch_size, embedding_size)
		# positive_key = torch.randn(batch_size, embedding_size)
		# negative_keys = torch.randn(batch_size, num_negative, embedding_size)
		# output = loss(query, positive_key, negative_keys)

		return x, loss


class ConvKB(nn.Module):
	def __init__(self, params):
		super(ConvKB, self).__init__()
		self.p = params

		self.config = {"convkb_drop_prob":0.5, "kernel_size":1,"hidden_size":200,
					   "num_of_filters":64,"use_init_embeddings":False, "lmbda":0.2,
					   "lmbda2":0.01}

		self.conv1_bn = nn.BatchNorm2d(1)
		self.conv_layer = nn.Conv2d(1, self.config["num_of_filters"], (self.config["kernel_size"], 3))  # kernel size x 3
		self.conv2_bn = nn.BatchNorm2d(self.config["num_of_filters"])
		self.dropout = nn.Dropout(self.config["convkb_drop_prob"])
		self.non_linearity = nn.ReLU()  # you should also tune with torch.tanh() or torch.nn.Tanh()
		self.fc_layer = nn.Linear((self.config["hidden_size"] - self.config["kernel_size"] + 1) * self.config["num_of_filters"], 1,
								  bias=False)

		self.criterion = nn.Softplus()
		# self.init_parameters()
		nn.init.xavier_uniform_(self.fc_layer.weight.data)
		nn.init.xavier_uniform_(self.conv_layer.weight.data)

	# def init_parameters(self):
	# 	if self.config["use_init_embeddings"] == False:
	# 		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
	# 		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
	#
	# 	else:
	# 		# self.ent_embeddings.weight.data = self.config.init_ent_embs
	# 		# self.rel_embeddings.weight.data = self.config.init_rel_embs
	# 		raise NotImplemented
	#
	#


	def _calc(self, h, r, t):
		h = h.unsqueeze(1)  # bs x 1 x dim
		r = r.unsqueeze(1)
		t = t.unsqueeze(1)

		conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
		conv_input = conv_input.transpose(1, 2)
		# To make tensor of size 4, where second dim is for input channels
		conv_input = conv_input.unsqueeze(1)
		conv_input = self.conv1_bn(conv_input)
		out_conv = self.conv_layer(conv_input)
		out_conv = self.conv2_bn(out_conv)
		out_conv = self.non_linearity(out_conv)
		out_conv = out_conv.view(-1, (self.config["hidden_size"] - self.config["kernel_size"] + 1) * self.config["num_of_filters"])
		input_fc = self.dropout(out_conv)
		score = self.fc_layer(input_fc).view(-1)

		return -score

	def loss(self, score, regul):
		return torch.mean(self.criterion(score * self.batch_y)) + self.config["lmbda"] * regul

	def forward(self, batch_h_e, batch_r_e, batch_t_e, batch_y):


		h = batch_h_e       # bs x dim
		r = batch_r_e
		t = batch_t_e
		self.batch_y = batch_y
		score = self._calc(h, r, t)

		# regularization
		l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
		for W in self.conv_layer.parameters():
			l2_reg = l2_reg + W.norm(2)
		for W in self.fc_layer.parameters():
			l2_reg = l2_reg + W.norm(2)

		return self.loss(score, l2_reg)



# class ConvKB(nn.Module):
#     """
#         In `A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network`_ (ConvKB),
#         each triple (head entity, relation, tail entity) is represented as a 3-column matrix where each column vector represents a triple element
#         Portion of the code based on daiquocnguyen_.
#         Args:
#             config (object): Model configuration parameters.
#         .. _daiquocnguyen:
#             https://github.com/daiquocnguyen/ConvKB
#         .. _A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network:
#             https://www.aclweb.org/anthology/N18-2053
#     """
#     def __init__(self, params):
#         super(ConvKB, self).__init__()
#         # param_list = ["tot_entity", "tot_relation", "hidden_size", "num_filters", "filter_sizes"]
#         # param_dict = self.load_params(param_list, kwargs)
#         # self.__dict__.update(param_dict)
#
# 		self.p = params
#
#         num_total_ent = self.tot_entity
#         num_total_rel = self.tot_relation
#         k = self.hidden_size
#         num_filters = self.num_filters
#         filter_sizes = self.filter_sizes
#         device = kwargs["device"]
#
#         self.ent_embeddings = nn.Embedding(self.)  # NamedEmbedding("ent_embedding", num_total_ent, k)
#         self.rel_embeddings = self.rel_embeddings                       # NamedEmbedding("rel_embedding", num_total_rel, k)
#         nn.init.xavier_uniform_(self.ent_embeddings.weight)
#         nn.init.xavier_uniform_(self.rel_embeddings.weight)
#
#         self.parameter_list = [
#             self.ent_embeddings,
#             self.rel_embeddings,
#         ]
#
#         self.conv_list = [nn.Conv2d(1, num_filters, (3, filter_size), stride=(1, 1)).to(device) for filter_size in filter_sizes]
#         conv_out_dim = num_filters*sum([(k-filter_size+1) for filter_size in filter_sizes])
#         self.fc1 = nn.Linear(in_features=conv_out_dim, out_features=1, bias=True)
#
#         self.loss = Criterion.pointwise_logistic
#
#     def embed(self, h, r, t):
#         """Function to get the embedding value.
#            Args:
#                h (Tensor): Head entities ids.
#                r (Tensor): Relation ids of the triple.
#                t (Tensor): Tail entity ids of the triple.
#             Returns:
#                 Tensors: Returns head, relation and tail embedding Tensors.
#         """
#         emb_h = self.ent_embeddings(h)
#         emb_r = self.rel_embeddings(r)
#         emb_t = self.ent_embeddings(t)
#         return emb_h, emb_r, emb_t
#
#     def forward(self, h_emb, r_emb, t_emb):
#         # h_emb, r_emb, t_emb = self.embed(h, r, t)
#         first_dimen = list(h_emb.shape)[0]
#
#         stacked_h = torch.unsqueeze(h_emb, dim=1)
#         stacked_r = torch.unsqueeze(r_emb, dim=1)
#         stacked_t = torch.unsqueeze(t_emb, dim=1)
#
#         stacked_hrt = torch.cat([stacked_h, stacked_r, stacked_t], dim=1)
#         stacked_hrt = torch.unsqueeze(stacked_hrt, dim=1)  # [b, 1, 3, k]
#
#         stacked_hrt = [conv_layer(stacked_hrt) for conv_layer in self.conv_list]
#         stacked_hrt = torch.cat(stacked_hrt, dim=3)
#         stacked_hrt = stacked_hrt.view(first_dimen, -1)
#         preds = self.fc1(stacked_hrt)
#         preds = torch.squeeze(preds, dim=-1)
#         return preds


class Way2(torch.nn.Module):
	def __init__(self, parmas, ent_embed, rel_embed):
		super(Way2, self).__init__()
		self.params = parmas
		self.ent_embed = ent_embed
		self.rel_embed = rel_embed
		self.kernelsize = (5,5)


		# -*-*-*- convolution -*-*-*-
		# self.cnn = torch.nn.Sequential(
		# 	torch.nn.Conv2d(1, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout(0.2),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	# torch.nn.Dropout(0.2),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout(0.2),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# )

		# -*-*-*- Depthwise separable convolution -*-*-*-
		self.cnn = torch.nn.Sequential(
			torch.nn.Conv2d(1, 32, self.kernelsize, stride=1, padding=0),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
			SeparableConv2d(32, 32, self.kernelsize, stride=1, padding=0),
			# torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
			SeparableConv2d(32, 32, self.kernelsize, stride=1, padding=0),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
			SeparableConv2d(32, 64, self.kernelsize, stride=1, padding=0),
			# torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			SeparableConv2d(64, 64, self.kernelsize, stride=1, padding=0),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
		)

		self.filtered_shape = (self.params.embed_dim-5*self.kernelsize[0]+5,
							   self.params.embed_dim-5*self.kernelsize[1]+5)
		self.way2_fc = torch.nn.Linear(64*self.filtered_shape[0]*self.filtered_shape[1], 1)

	def forward(self, ent_target_emb, obj_emb):
		"""
		Parameters
		----------
		ent_target_emb: 
		obj_emb: 
		Returns
		-------
		"""
		inp = torch.bmm(obj_emb.unsqueeze(2), ent_target_emb.unsqueeze(1))   
		stack_inp = inp.unsqueeze(1)      
		x = self.cnn(stack_inp)      
		x = x.flatten(1)
		x = self.way2_fc(x)          
		score = x.squeeze(1)             
		score = F.sigmoid(score)
		return score



class InteractE(torch.nn.Module):
	"""
	Parameters
	----------
	params:        	Hyperparameters of the model
	chequer_perm:   Reshaping to be used by the model
	
	Returns
	-------
	The InteractE model instance
		
	"""
	def __init__(self, params, chequer_perm, checquer_perm_3vec):
		super(InteractE, self).__init__()
		self.p                  = params
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.inp_drop = torch.nn.Dropout(self.p.inp_drop)
		self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
		self.feature_map_drop = torch.nn.Dropout2d(self.p.feat_drop)
		self.bn0 = torch.nn.BatchNorm2d(self.p.perm).to(self.device)
		# self.conv1_bn = nn.BatchNorm2d(1)
		# self.ent_embed = torch.nn.Embedding(self.p.num_ent,elf.p.embed_dim, padding_idx=None).to(self.device)
		# xavier_normal_(self.ent_embed.weight)       #

		# self.criter = nn.Softplus()
		# self.use_name = "check"
		#
		self.rel_embed		= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None).to(self.device); xavier_normal_(self.rel_embed.weight)
		# self.register_parameter()
		self.ent_embed      = nn.Parameter(self.get_pretrained_embed('./')).to(self.device)

		self.bceloss		= torch.nn.BCELoss()
		self.bprloss = BPRLoss()

		self.way1_bceloss = BinaryCrossEntropyLoss(self.p)

		flat_sz_h 		= self.p.k_h
		flat_sz_w 		= 2*self.p.k_w
		self.padding 		= 0

		self.bn1 		= torch.nn.BatchNorm2d(self.p.num_filt*self.p.perm).to(self.device)
		self.flat_sz 		= flat_sz_h * flat_sz_w * self.p.num_filt*self.p.perm

		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim).to(self.device)
		self.fc 		= torch.nn.Linear(self.flat_sz, self.p.embed_dim).to(self.device)
		self.chequer_perm	= chequer_perm       
		self.chequer_perm_3vec = checquer_perm_3vec   

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent).to(self.device)))
		self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.num_filt, 1, self.p.ker_sz,  self.p.ker_sz).to(self.device)));
		xavier_normal_(self.conv_filt)

		nin1, nout1, kernel1, pad1 = self.p.perm, 32, 5, 0
		nin2, nout2, kernel2, pad2 = 1, 16, 5, 0
		self.dpsconv1 = depthwise_separable_conv(nin=nin1, nout=nout1, kernel_size=kernel1, padding=pad1, bias=False).to(self.device)
		self.dpsconv2 = depthwise_separable_conv(nin=nin2, nout=nout2, kernel_size=kernel2, padding=pad2, bias=False).to(self.device)
		self.fc_fus1 = nn.Linear(self.p.embed_dim*2,self.p.embed_dim).to(self.device)
		self.h1, self.w1 = np.sqrt(self.p.embed_dim*2), np.sqrt(self.p.embed_dim*2)
		sz1 = (self.w1-kernel1+1) * (self.h1-kernel1+1) * nout1
		self.h2 = 20
		self.w2 = int(self.p.embed_dim / self.h2)
		sz2 = (self.w2-kernel2+1) * (self.h2-kernel2+1) * nout2
		self.fc_fus2 = nn.Linear(int(sz1+sz2), self.p.embed_dim).to(self.device)


		self.kgml = KGML(self.p).to(self.device)
		self.way1 = Way1(self.p, self.chequer_perm_3vec).to(self.device)
		self.tc = ConvKB(self.p).to(self.device)

	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0];      
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def way2_loss(self, pos_score, neg_score):
		loss = self.bprloss(pos_score, neg_score)
		return loss

	def way1_loss(self, pos_trips_pred, neg_trips_pred):
		loss = self.way1_bceloss(pos_trips_pred, neg_trips_pred)
		return loss

	def circular_padding_chw(self, batch, padding):
		upper_pad	= batch[..., -padding:, :]
		lower_pad	= batch[..., :padding, :]
		temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

		left_pad	= temp[..., -padding:]
		right_pad	= temp[..., :padding]
		padded		= torch.cat([left_pad, temp, right_pad], dim=3)
		return padded

	def get_pretrained_embed(self, path):
		res = []
		with open(os.path.join(path, 'sen.vec'), 'r') as f:
			for line in f.readlines():
				line = line.strip()
				line = line[1:-1]
				line = line.replace("'", "").replace(" ", "").split(',')
				line = list(map(float, line))
				res.append(line)
		res = torch.tensor(res)
		return res


	def forward(self, sub, rel, neg_ents, label=None, is_train:bool=True, all_triples=None, sr2o_all=None, so2r=None, strategy='one_to_x', step=None):
		bs = sub.shape[0]
		sub_emb		= self.ent_embed(sub)  
		rel_emb		= self.rel_embed(rel)  
		if is_train:
			pass
		comb_emb	= torch.cat([sub_emb, rel_emb], dim=1)  
		chequer_perm	= comb_emb[:, self.chequer_perm]  
		stack_inp	= chequer_perm.reshape((-1, self.p.perm, 2*self.p.k_w, self.p.k_h))

		stack_inp	= self.bn0(stack_inp)
		x		= self.inp_drop(stack_inp)
		x		= self.circular_padding_chw(x, self.p.ker_sz//2)   
		x		= F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding, groups=self.p.perm)  


		x		= self.bn1(x)  
		x		= F.relu(x)
		x		= self.feature_map_drop(x)
		x		= x.view(-1, self.flat_sz)   
		x		= self.fc(x)   
		x		= self.hidden_drop(x)
		x		= self.bn2(x)
		x		= F.relu(x)


		if strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
			x += self.bias[neg_ents]

		pred	= torch.sigmoid(x)


		way2_pos_score, way2_neg_score = 0, 0
		way2Loss = 0
		way1Loss = 0
		if is_train:
			label = (label>0.5).byte()
			label_idxes = torch.nonzero(label==1)
			temp_label_dict = defaultdict(list)
			for i in label_idxes:
				temp_label_dict[i[0].item()].append(i[1].item())
			obj_lists = list(map(lambda x: x[1], temp_label_dict.items()))
			obj = torch.tensor(list(map(lambda x: random.choice(x), obj_lists))).to(self.device)

			assert torch.numel(obj) == len(sub)
			# # just for DEBUG

			r_lbs = []
			temp_rel = []
			temp_neg_rel = []
			temp_sub = []
			temp_obj = []
			record, cnt = [], 0
			st2 = datetime.now()
			for idx,lst in enumerate(obj_lists):
				for o in lst:
					r_label = torch.tensor(so2r[(sub[idx].item(), o)]).unsqueeze(0)
					"""
					try:
						r_label = torch.tensor(so2r[(sub[idx].item(), o)]).unsqueeze(0)
					except KeyError:
						# tmp = torch.tensor(so2r[(o, sub[idx].item())])
						# r_label = torch.where(tmp<self.p.num_rel, tmp, tmp-self.p.num_rel).unsqueeze(0)
						# # r_label = (torch.tensor(so2r[(o, sub[idx].item())])-self.p.num_rel).unsqueeze(0)
						raise KeyError
					"""
					target = torch.zeros(r_label.shape[0], self.p.num_rel).scatter(1, r_label, 1)
					r_lbs.append(target)

					temp_sub.append(sub[idx])

					temp_neg_rel.append(list(np.nonzero(target-1)[0]))    # 取负的
					temp_obj.append(o)
					cnt += 1
				record.append(cnt)


			r_lbs = torch.cat(r_lbs, dim=0).to(self.device)
			temp_rel =list(r_lbs.argmax(1))
			if self.p.lbl_smooth != 0.0:
				r_lbs = (1-self.p.lbl_smooth)*r_lbs + self.p.lbl_smooth/self.p.num_rel
			rp_task_sub_emb = self.ent_embed(torch.LongTensor(temp_sub).to(self.device))
			rp_task_obj_emb = self.ent_embed(torch.LongTensor(temp_obj).to(self.device))
			rp_task_rel_emb = self.ent_embed(torch.LongTensor(temp_rel).to(self.device))
			_, way2Loss = self.kgml(rp_task_sub_emb, rp_task_obj_emb, r_lbs)
			obj_emb = self.ent_embed(obj)
			samples_idxes = get_samples_idxes(sub, pred, label, obj, device=self.device).detach().data
			neg_obj_list = []
			n_neg = self.p.need_n_neg
			start = datetime.now()


			for idx, row in enumerate(samples_idxes):
				neg_obj_temp = []
				n = 0
				for ele in row:
					flags = torch.tensor(
						(sub[idx].cpu().item(), rel[idx].cpu().item(), ele.cpu().item())) == torch.tensor(all_triples)
					flag = flags[:, 0] & flags[:, 1] & flags[:, 2]
					if not flag.any().item():  # if not in KG
						neg_obj_temp.append(ele.cpu().item())
						n += 1
						if len(neg_obj_temp) == n_neg:
							break
				neg_obj_list.append(neg_obj_temp)  # neg

			# prepare query positive negative
			need_rel_len = min(n_neg, self.p.num_rel)
			temp_sub_aug_positive = temp_sub
			temp_rel_aug_positive = list(r_lbs.argmin(1))
			temp_obj_aug_positive = temp_obj
			# negative
			temp_sub = [ele.item() for ele in temp_sub]
			temp_sub_neg = list(zip(*[temp_sub for _ in range(need_rel_len)]))
			list2select = np.array([list(range(self.p.num_rel)) for _ in range(len(temp_sub))])
			temp_rel_neg = [np.delete(l, temp_rel[idx].item())[:need_rel_len] for idx,l in enumerate(list2select)]
			temp_obj_neg = []
			aux = list(np.diff(np.array(record)))
			aux = [record[0] -0] + aux
			for idx,j in enumerate(aux):
				for _ in range(j):
					temp_obj_neg.append([random.choice(neg_obj_list[idx])] * need_rel_len)

			# triplet loss
			# anchor = rp_task_sub_emb * rp_task_rel_emb * rp_task_obj_emb        # (batch_size, embedding_size)
			# print(f"anchor shape: {anchor.shape}")
			# positive = self.ent_embed(torch.LongTensor(temp_sub_aug_positive).to(self.device)) * self.rel_embed(
			# 	torch.LongTensor(temp_rel_aug_positive).to(self.device)) * self.ent_embed(
			# 	torch.LongTensor(temp_obj_aug_positive).to(self.device))  # (batch_size, embedding_size)
			# print(f"positive shape: {positive.shape}")
			# negative = self.ent_embed(torch.LongTensor(temp_sub_neg).to(self.device)) * self.rel_embed(
			# 	torch.LongTensor(temp_rel_neg).to(self.device))* self.ent_embed(
			# 	torch.LongTensor(temp_obj_neg).to(self.device))  # (batch_size, embedding_size)
			# print(f"negative shape: {negative.shape}")
			# tripletloss = torch.nn.functional.triplet_margin_loss(anchor, positive, negative, reduction='mean')
			# way2Loss += tripletloss

			# infonce
			loss = InfoNCE(negative_mode='paired')
			query = rp_task_sub_emb * rp_task_rel_emb * rp_task_obj_emb        # (batch_size, embedding_size)
			positive_key = self.ent_embed(torch.LongTensor(temp_sub_aug_positive).to(self.device)) * self.rel_embed(torch.LongTensor(temp_rel_aug_positive).to(self.device)) * self.ent_embed(torch.LongTensor(temp_obj_aug_positive).to(self.device))
			negative_keys = self.ent_embed(torch.LongTensor(temp_sub_neg).to(self.device)) * self.rel_embed(
					torch.LongTensor(temp_rel_neg).to(self.device))* self.ent_embed(
					torch.LongTensor(temp_obj_neg).to(self.device))
			infonceloss = loss(query, positive_key, negative_keys)
			way2Loss += infonceloss





		if is_train:

			neg_obj_list = list(zip(*neg_obj_list))
			tc_s, tc_r, tc_o = [], [], []
			y = []

			tc_s.append(sub_emb)
			tc_r.append(rel_emb)
			tc_o.append(obj_emb)
			y.extend([1 for _ in range(sub_emb.shape[0])])

			for i in range(n_neg):
				tc_s.append(sub_emb)
				tc_r.append(rel_emb)
				neg_obj_i_emb = self.ent_embed(torch.tensor(neg_obj_list[i]).to(self.device))
				tc_o.append(neg_obj_i_emb)
				y.extend([0 for _ in range(sub_emb.shape[0])])


			for idx,iteml in enumerate(temp_neg_rel):
				new_temp_sub = [temp_sub[idx]] * len(iteml)
				new_temp_obj = [temp_obj[idx]] * len(iteml)
				new_temp_sub_emb = self.ent_embed(torch.LongTensor(new_temp_sub).to(self.device))
				new_temp_obj_emb = self.ent_embed(torch.LongTensor(new_temp_obj).to(self.device))
				new_temp_rel_emb = self.rel_embed(torch.LongTensor(iteml).to(self.device))
				tc_s.append(new_temp_sub_emb)
				tc_r.append(new_temp_rel_emb)
				tc_o.append(new_temp_obj_emb)
				y.extend([0 for _ in range(new_temp_sub_emb.shape[0])])


			# adding more
			z1 = self.dpsconv1(stack_inp)   # Size([bs, 32, 16, 16])
			z1 = torch.flatten(z1, start_dim=1)    # Size([bs, 32*16*16])

			z2 = self.fc_fus1(comb_emb)   # Size([bs, dim])
			z2 = z2.view(-1, 1, self.h2, self.w2)   # Size([bs, 1, h2, w2])

			z2 = self.dpsconv2(z2)
			z2 = torch.flatten(z2, start_dim=1)

			z = self.fc_fus2(torch.cat([z1, z2], dim=1))   # Size([bs, dim])

			tc_s.append(z)
			tc_r.append(z)
			tc_o.append(obj_emb)
			y.extend([1 for _ in range(z.shape[0])])



			new_tc_se = torch.cat(tc_s, dim=0)       # new batchsize x dim
			new_tc_re = torch.cat(tc_r, dim=0)
			new_tc_oe = torch.cat(tc_o, dim=0)
			y = torch.LongTensor(y).to(self.device)


			tcloss = self.tc(new_tc_se, new_tc_re, new_tc_oe, y)

			way1Loss = tcloss






		# if is_train:
		# 	neg_obj_list = list(zip(*neg_obj_list))
		# 	pos_out = self.way1(bs, sub_emb, rel_emb, obj_emb)
		# 	temp_losses = []
		# 	for i in range(n_neg):
		# 		neg_obj_i_emb = self.ent_embed(torch.tensor(neg_obj_list[i]).to(self.device))
		# 		neg_out = self.way1(bs, sub_emb, rel_emb, neg_obj_i_emb)
		# 		temp_loss = self.way1_bceloss(pos_out, neg_out).item()
		# 		temp_losses.append(temp_loss)
		#
		# 	way1Loss = torch.tensor(temp_losses).mean()
		#
		# torch.cuda.empty_cache()

		return pred, way2Loss, way1Loss
