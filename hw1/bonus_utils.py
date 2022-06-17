import argparse
import torch
from torch.utils.data import Dataset , DataLoader

class Dataset(Dataset):
	def __init__(self , data , label , mask):
		self.data = data
		self.label = label
		self.mask = mask

	def __len__(self):
		return len(self.data)

	def __getitem__(self , index):
		if self.label is not None:
			return torch.tensor(self.data[index]) , torch.tensor(self.label[index]) , torch.tensor(self.mask[index])
		else:
			return torch.tensor(self.data[index]) , torch.tensor(self.mask[index])

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data' , type = str , default = '../data/slot/train.json')
	parser.add_argument('--validation_data' , type = str , default = '../data/slot/eval.json')
	parser.add_argument('--test_data' , type = str , default = '../data/slot/test.json')
	parser.add_argument('--output_file' , type = str , default = 'prediction.csv')
	parser.add_argument('--checkpoint_directory' , type = str , default = './')
	parser.add_argument('--checkpoint' , type = str , default = 'slot_tagger_bonus.pth')
	parser.add_argument('--max_length' , type = int , default = 40)
	parser.add_argument('--batch_size' , type = int , default = 128)
	parser.add_argument('--learning_rate' , type = float , default = 0.002)
	parser.add_argument('--epoch' , type = int , default = 50)
	args = parser.parse_args()
	return args

def get_dataloader(data , label , mask , mode , batch_size = 1024 , num_workers = 8):
	dataset = Dataset(data , label , mask)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss = None , accuracy = None):
	if batch < total_batch:
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '')
	else:
		data = total_data
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f} , train accuracy : {:.5f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss , accuracy))