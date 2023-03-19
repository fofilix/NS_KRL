import torch.nn

from helper import *

class CrossE(torch.nn.Module):
	"""
	Proposed method in the paper. Refer Section 6 of the paper for mode details 

	Parameters
	----------
	params:        	Hyperparameters of the model
	chequer_perm:   Reshaping to be used by the model
	
	Returns
	-------
	The InteractE model instance
		
	"""
	def __init__(self, params, chequer_perm):
		super(CrossE, self).__init__()

		self.p                  = params
		self.ent_embed		= torch.nn.Embedding(self.p.num_ent,   self.p.embed_dim, padding_idx=None); xavier_normal_(self.ent_embed.weight)
		self.rel_embed		= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None); xavier_normal_(self.rel_embed.weight)
		self.C = torch.nn.Embedding(self.p.num_rel*2,   self.p.embed_dim, padding_idx=None); xavier_normal_(self.ent_embed.weight)

		self.bceloss		= torch.nn.BCELoss()

		self.register_parameter('bias0', Parameter(torch.zeros(self.p.embed_dim)))
		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0];      
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def forward(self, sub, rel, neg_ents, strategy='one_to_x'):

		sub_emb		= self.ent_embed(sub)  
		rel_emb		= self.rel_embed(rel)  
		c_emb = self.C(rel)        

		sub_I = c_emb * sub_emb    
		rel_I = sub_I * rel_emb    

		tmp_I = sub_I + rel_I
		tmp_I +=  self.bias0.expand_as(tmp_I)
		q_sr = F.tanh(tmp_I)       
		x = q_sr


		if strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
			x += self.bias[neg_ents]

		pred	= torch.sigmoid(x)

		return pred
