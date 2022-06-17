import argparse
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data' , type = str , default = 'data/train.jsonl')
	parser.add_argument('--validation_data' , type = str , default = 'data/public.jsonl')
	parser.add_argument('--test_data' , type = str , default = 'data/public.jsonl')
	parser.add_argument('--output_file' , type = str , default = 'prediction.jsonl')
	parser.add_argument('--model_name' , type = str , default = 'google/mt5-small')
	parser.add_argument('--checkpoint_name' , type = str , default = 'checkpoint/')
	parser.add_argument('--task_prefix' , type = str , default = '')
	parser.add_argument('--max_length_of_source' , type = int , default = 256)
	parser.add_argument('--max_length_of_target' , type = int , default = 64)
	parser.add_argument('--batch_size' , type = int , default = 16)
	parser.add_argument('--accum_iter' , type = int , default = 8)
	parser.add_argument('--learning_rate' , type = float , default = 0.0005)
	parser.add_argument('--weight_decay' , type = float , default = 0)
	parser.add_argument('--epoch' , type = int , default = 15)
	parser.add_argument('--num_beams' , type = int , default = 4)
	parser.add_argument('--do_sample' , type = int , default = 0)
	parser.add_argument('--top_k' , type = int , default = 50)
	parser.add_argument('--top_p' , type = float , default = 0.9)
	parser.add_argument('--temperature' , type = float , default = 10.0)
	parser.add_argument('--ratio' , type = float , default = 0.1)
	parser.add_argument('--plot' , type = int , default = 0)
	args = parser.parse_args()
	return args

class Dataset(Dataset):
	def __init__(self , data , baseline , tokenizer , task_prefix , max_length_of_source , max_length_of_target , mode):
		self.data = data
		self.baseline = baseline
		self.tokenizer = tokenizer
		self.task_prefix = task_prefix
		self.max_length_of_source = max_length_of_source
		self.max_length_of_target = max_length_of_target
		self.mode = mode

	def __len__(self):
		return len(self.data)

	def __getitem__(self , index):
		if self.mode != 'test':
			maintext = self.tokenizer(self.task_prefix + self.data[index]['maintext'] , max_length = self.max_length_of_source , truncation = True , padding = 'max_length')
			title = self.tokenizer(self.data[index]['title'] , max_length = self.max_length_of_target , truncation = True , padding = 'max_length')
			input_ids = maintext.input_ids
			attention_mask = maintext.attention_mask
			labels = title.input_ids
			labels[labels == self.tokenizer.pad_token_id] = -100
			if self.baseline is None:
				return torch.tensor(input_ids) , torch.tensor(attention_mask) , torch.tensor(labels) , self.data[index]['title']
			else:
				return torch.tensor(input_ids) , torch.tensor(attention_mask) , torch.tensor(labels) , self.data[index]['title'] , self.baseline[index]
		else:
			maintext = self.tokenizer(self.task_prefix + self.data[index]['maintext'] , max_length = self.max_length_of_source , truncation = True , padding = 'max_length')
			input_ids = maintext.input_ids
			attention_mask = maintext.attention_mask
			return torch.tensor(input_ids) , torch.tensor(attention_mask)

def get_dataloader(data , baseline , tokenizer , args , mode):
	dataset = Dataset(data , baseline , tokenizer , args.task_prefix , args.max_length_of_source , args.max_length_of_target , mode)
	dataloader = DataLoader(dataset , batch_size = args.batch_size // args.accum_iter , shuffle = (mode == 'train') , num_workers = 8)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss = None , ROUGE_1 = None , ROUGE_2 = None , ROUGE_l = None):
	if batch < total_batch:
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}] ({}s)'.format(epoch , total_epoch , data , total_data , bar , total_time) , end = '')
	else:
		data = total_data
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f} , train ROUGE-1 : {:.1f} , train ROUGE-2 : {:.1f} , train ROUGE-L : {:.1f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss , 100 * ROUGE_1 , 100 * ROUGE_2 , 100 * ROUGE_l))

def plot_learning_curve(train_loss_list , validation_loss_list , train_ROUGE_list , validation_ROUGE_list):
	import matplotlib.pyplot as plt

	plt.plot(np.arange(1 , len(train_loss_list) + 1) , train_loss_list , marker = 'o' , label = 'Train')
	plt.plot(np.arange(1 , len(validation_loss_list) + 1) , validation_loss_list , marker = 'o' , label = 'Validation')
	plt.title('Learning Curve')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.xticks(np.arange(1 , len(train_loss_list) + 1))
	plt.legend()
	plt.savefig('loss.png')
	plt.clf()

	plt.plot(np.arange(1 , len(train_ROUGE_list) + 1) , [train_ROUGE_list[i]['ROUGE-1'] for i in range(len(train_ROUGE_list))] , marker = 'o' , label = 'Train')
	plt.plot(np.arange(1 , len(validation_ROUGE_list) + 1) , [100 * validation_ROUGE_list[i]['ROUGE-1'] for i in range(len(validation_ROUGE_list))] , marker = 'o' , label = 'Validation')
	plt.title('Learning Curve')
	plt.xlabel('Epoch')
	plt.ylabel('ROUGE-1')
	plt.xticks(np.arange(1 , len(train_ROUGE_list) + 1))
	plt.legend()
	plt.savefig('ROUGE_1.png')
	plt.clf()

	plt.plot(np.arange(1 , len(train_ROUGE_list) + 1) , [train_ROUGE_list[i]['ROUGE-2'] for i in range(len(train_ROUGE_list))] , marker = 'o' , label = 'Train')
	plt.plot(np.arange(1 , len(validation_ROUGE_list) + 1) , [100 * validation_ROUGE_list[i]['ROUGE-2'] for i in range(len(validation_ROUGE_list))] , marker = 'o' , label = 'Validation')
	plt.title('Learning Curve')
	plt.xlabel('Epoch')
	plt.ylabel('ROUGE-2')
	plt.xticks(np.arange(1 , len(train_ROUGE_list) + 1))
	plt.legend()
	plt.savefig('ROUGE_2.png')
	plt.clf()

	plt.plot(np.arange(1 , len(train_ROUGE_list) + 1) , [train_ROUGE_list[i]['ROUGE-L'] for i in range(len(train_ROUGE_list))] , marker = 'o' , label = 'Train')
	plt.plot(np.arange(1 , len(validation_ROUGE_list) + 1) , [100 * validation_ROUGE_list[i]['ROUGE-L'] for i in range(len(validation_ROUGE_list))] , marker = 'o' , label = 'Validation')
	plt.title('Learning Curve')
	plt.xlabel('Epoch')
	plt.ylabel('ROUGE-L')
	plt.xticks(np.arange(1 , len(train_ROUGE_list) + 1))
	plt.legend()
	plt.savefig('ROUGE_L.png')
	plt.clf()