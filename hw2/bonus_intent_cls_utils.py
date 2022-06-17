import argparse
import torch
from torch.utils.data import Dataset , DataLoader

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data' , type = str , default = '../hw1/data/intent/train.json')
	parser.add_argument('--validation_data' , type = str , default = '../hw1/data/intent/eval.json')
	parser.add_argument('--test_data' , type = str , default = '../hw1/data/intent/test.json')
	parser.add_argument('--output_file' , type = str , default = 'prediction.csv')
	args = parser.parse_args()
	return args

def load_config():
	config = {
		'model_name'      : 'bert-base-uncased' ,
		'checkpoint_root' : './' , 
		'checkpoint_name' : 'IntentClassification/' , 
		'max_length'      : 512 , 
		'batch_size'      : 8 , 
		'accum_iter'      : 2 , 
		'learning_rate'   : 0.00005 , 
		'weight_decay'    : 0.01 , 
		'epoch'           : 5
	}

	return config

class Dataset(Dataset):
	def __init__(self , data , tokenizer , intent2label , config , mode):
		self.data = data
		self.tokenizer = tokenizer
		self.intent2label = intent2label
		self.max_length = config['max_length']
		self.mode = mode

	def __len__(self):
		return len(self.data)

	def __getitem__(self , index):
		text = self.data[index]['text']
		input_ids = self.tokenizer.encode(text , max_length = self.max_length , truncation = True , add_special_tokens = True)
		token_type_ids = self.max_length * [1]
		attention_mask = len(input_ids) * [1] + (self.max_length - len(input_ids)) * [0]
		input_ids += max(0 , self.max_length - len(input_ids)) * [self.tokenizer.pad_token_id]

		if self.mode != 'test':
			return torch.tensor(input_ids) , torch.tensor(token_type_ids) , torch.tensor(attention_mask) , self.intent2label[self.data[index]['intent']]
		else:
			return torch.tensor(input_ids) , torch.tensor(token_type_ids) , torch.tensor(attention_mask)

def get_dataloader(data , tokenizer , intent2label , config , mode):
	dataset = Dataset(data , tokenizer , intent2label , config , mode)
	dataloader = DataLoader(dataset , batch_size = config['batch_size'] // config['accum_iter'] , shuffle = (mode == 'train') , num_workers = 8)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss = None , accuracy = None):
	if batch < total_batch:
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}] ({}s)'.format(epoch , total_epoch , data , total_data , bar , total_time) , end = '')
	else:
		data = total_data
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f} , train accuracy : {:.5f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss , accuracy))