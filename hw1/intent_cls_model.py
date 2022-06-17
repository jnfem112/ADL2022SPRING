import torch
import torch.nn as nn

class Classifier(nn.Module):
	def __init__(self , WordVector):
		super(Classifier , self).__init__()

		embedding_weight = torch.FloatTensor([WordVector[i] for i in range(len(WordVector))])
		self.embedding = nn.Embedding(embedding_weight.size(dim = 0) , embedding_weight.size(dim = 1) , padding_idx = WordVector.get_index('<PAD>'))
		self.embedding.weight = nn.Parameter(embedding_weight)
		self.embedding.weight.requires_grad = True

		self.linear_1 = nn.Linear(embedding_weight.size(dim = 1) , 512)
		self.self_attention_layer = nn.TransformerEncoderLayer(d_model = 512 , nhead = 4 , dim_feedforward = 512 , dropout = 0.1 , activation = 'relu' , batch_first = True)
		self.self_attention = nn.TransformerEncoder(self.self_attention_layer , num_layers = 2)

		self.linear_2 = nn.Sequential(
			nn.Linear(1536 , 2048 , bias = True) ,
			nn.BatchNorm1d(2048) , 
			nn.ReLU() ,
			nn.Dropout(0.5) , 
			nn.Linear(2048 , 150 , bias = True)
		)

	def forward(self , x):
		x = self.embedding(x)
		x = self.linear_1(x)
		x = self.self_attention(x)
		x = torch.cat([x.min(dim = 1).values , x.max(dim = 1).values , x.mean(dim = 1)] , dim = 1)
		x = self.linear_2(x)
		return x