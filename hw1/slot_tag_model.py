import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
	def __init__(self , WordVector):
		super(Classifier , self).__init__()

		embedding_weight = torch.FloatTensor([WordVector[i] for i in range(len(WordVector))])
		self.embedding = nn.Embedding(embedding_weight.size(dim = 0) , embedding_weight.size(dim = 1) , padding_idx = WordVector.get_index('<PAD>'))
		self.embedding.weight = nn.Parameter(embedding_weight)
		self.embedding.weight.requires_grad = True

		self.linear_1 = nn.Linear(embedding_weight.size(dim = 1) , 512)
		self.recurrent = nn.LSTM(512 , 1024 , batch_first = True , bias = True , num_layers = 4 , dropout = 0.3 , bidirectional = True)

		self.linear_2 = nn.Sequential(
			nn.Linear(2048 , 2048 , bias = True) ,
			nn.BatchNorm1d(2048) ,
			nn.ReLU() ,
			nn.Dropout(0.5) ,
			nn.Linear(2048 , 10 , bias = True)
		)

	def forward(self , x):
		x = self.embedding(x)
		x = self.linear_1(x)
		x , _ = self.recurrent(x)
		batch_size , seq_len , hidden_dim = x.shape
		x = x.contiguous().view(batch_size * seq_len , hidden_dim)
		x = self.linear_2(x)
		x = x.contiguous().view(batch_size , seq_len , 10)
		return x