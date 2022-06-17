import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--context_file' , type = str , default = 'data/context.json')
	parser.add_argument('--train_data' , type = str , default = 'data/train.json')
	parser.add_argument('--validation_data' , type = str , default = 'data/valid.json')
	parser.add_argument('--test_data' , type = str , default = 'data/test.json')
	parser.add_argument('--output_file' , type = str , default = 'prediction.csv')
	parser.add_argument('--base_model' , type = int , default = 0)
	parser.add_argument('--pretrained' , type = int , default = 1)
	parser.add_argument('--plot' , type = int , default = 0)
	args = parser.parse_args()
	return args

def load_config(base_model):
	config_1 = {
		'model_name'              : 'bert-base-chinese' if base_model else 'hfl/chinese-macbert-base' ,
		'checkpoint_root'         : './' , 
		'checkpoint_name'         : 'MultipleChoice/' , 
		'max_length_of_question'  : 50 ,  
		'max_length_of_paragraph' : 450 , 
		'max_length_of_sequence'  : 512 , 
		'batch_size'              : 16 , 
		'accum_iter'              : 16 , 
		'learning_rate'           : 0.00005 , 
		'weight_decay'            : 0.01 , 
		'epoch'                   : 5
	}

	config_2 = {
		'model_name'              : 'bert-base-chinese' if base_model else 'hfl/chinese-macbert-large' , 
		'checkpoint_root'         : './' , 
		'checkpoint_name'         : 'QuestionAnswering/' , 
		'max_length_of_question'  : 50 ,  
		'max_length_of_paragraph' : 450 , 
		'max_length_of_sequence'  : 512 , 
		'doc_stride'              : 50 , 
		'batch_size'              : 16 , 
		'accum_iter'              : 16 , 
		'learning_rate'           : 0.00005 , 
		'weight_decay'            : 0.01 , 
		'epoch'                   : 5
	}

	return config_1 , config_2

class MultipleChoiceDataset(Dataset):
	def __init__(self , context , data , tokenizer , config , mode):
		self.context = context
		self.data = data
		self.tokenizer = tokenizer
		self.max_length_of_question = config['max_length_of_question']
		self.max_length_of_paragraph = config['max_length_of_paragraph']
		self.max_length_of_sequence = config['max_length_of_sequence']
		self.mode = mode

	def __len__(self):
		return len(self.data)

	def __getitem__(self , index):
		question = self.data[index]['question']
		question_input_ids = self.tokenizer.encode(question , max_length = self.max_length_of_sequence , truncation = True , add_special_tokens = False)
		question_input_ids = question_input_ids[ : min(len(question_input_ids) , self.max_length_of_question)]

		paragraph_input_ids_list , label = [] , None
		for i , paragraph_index in enumerate(self.data[index]['paragraphs']):
			paragraph = self.context[paragraph_index]
			paragraph_input_ids = self.tokenizer.encode(paragraph , max_length = self.max_length_of_sequence , truncation = True , add_special_tokens = False)
			paragraph_input_ids = paragraph_input_ids[ : min(len(paragraph_input_ids) , self.max_length_of_paragraph)]
			paragraph_input_ids_list.append(paragraph_input_ids)
			if self.mode != 'test' and paragraph_index == self.data[index]['relevant']:
				label = i

		input_ids_list , token_type_ids_list , attention_mask_list = [] , [] , []
		for paragraph_input_ids in paragraph_input_ids_list:
			token_type_ids = (len(question_input_ids) + 2) * [0] + (self.max_length_of_sequence - len(question_input_ids) - 2) * [1]
			input_ids = [self.tokenizer.cls_token_id] + question_input_ids + [self.tokenizer.sep_token_id] + paragraph_input_ids + [self.tokenizer.sep_token_id]
			attention_mask = len(input_ids) * [1] + max(0 , self.max_length_of_sequence - len(input_ids)) * [0]
			input_ids += max(0 , self.max_length_of_sequence - len(input_ids)) * [self.tokenizer.pad_token_id]
			input_ids_list.append(input_ids)
			token_type_ids_list.append(token_type_ids)
			attention_mask_list.append(attention_mask)
		
		if self.mode != 'test':
			return torch.tensor(input_ids_list) , torch.tensor(token_type_ids_list) , torch.tensor(attention_mask_list) , label
		else:
			return torch.tensor(input_ids_list) , torch.tensor(token_type_ids_list) , torch.tensor(attention_mask_list)

class QuestionAnsweringDataset(Dataset):
	def __init__(self , context , data , tokenizer , config , mode):
		self.context = context
		self.data = data
		self.tokenizer = tokenizer
		self.max_length_of_question = config['max_length_of_question']
		self.max_length_of_paragraph = config['max_length_of_paragraph']
		self.max_length_of_sequence = config['max_length_of_sequence']
		self.doc_stride = config['doc_stride']
		self.mode = mode

	def __len__(self):
		return len(self.data)

	def __getitem__(self , index):
		if self.mode == 'train':
			question = self.data[index]['question']
			question_input_ids = self.tokenizer.encode(question , truncation = False , add_special_tokens = False)
			question_input_ids = question_input_ids[ : min(len(question_input_ids) , self.max_length_of_question)]

			paragraph = self.context[self.data[index]['relevant']]
			tokenized_paragraph = self.tokenizer(paragraph , truncation = False , add_special_tokens = False)
			paragraph_input_ids = tokenized_paragraph.input_ids
			answer_start_position = tokenized_paragraph.char_to_token(self.data[index]['answer']['start'])
			answer_end_position = tokenized_paragraph.char_to_token(self.data[index]['answer']['start'] + len(self.data[index]['answer']['text']) - 1)
			paragraph_start_position = random.randint(max(0 , answer_end_position - self.max_length_of_paragraph + 1) , answer_start_position)
			paragraph_end_position = min(paragraph_start_position + self.max_length_of_paragraph - 1 , len(paragraph_input_ids) - 1)
			paragraph_input_ids = paragraph_input_ids[paragraph_start_position : paragraph_end_position + 1]

			token_type_ids = (len(question_input_ids) + 2) * [0] + (self.max_length_of_sequence - len(question_input_ids) - 2) * [1]
			input_ids = [self.tokenizer.cls_token_id] + question_input_ids + [self.tokenizer.sep_token_id] + paragraph_input_ids + [self.tokenizer.sep_token_id]
			attention_mask = len(input_ids) * [1] + max(0 , self.max_length_of_sequence - len(input_ids)) * [0]
			input_ids += max(0 , self.max_length_of_sequence - len(input_ids)) * [self.tokenizer.pad_token_id]

			answer_start_position = (len(question_input_ids) + 2) + answer_start_position - paragraph_start_position
			answer_end_position = (len(question_input_ids) + 2) + answer_end_position - paragraph_start_position

			return torch.tensor(input_ids) , torch.tensor(token_type_ids) , torch.tensor(attention_mask) , answer_start_position , answer_end_position , self.data[index]['answer']['text']
		else:
			question = self.data[index]['question']
			question_input_ids = self.tokenizer.encode(question , truncation = False , add_special_tokens = False)
			question_input_ids = question_input_ids[ : min(len(question_input_ids) , self.max_length_of_question)]

			paragraph = self.context[self.data[index]['relevant']]
			paragraph_input_ids = self.tokenizer.encode(paragraph , truncation = False , add_special_tokens = False)
			input_ids_list , token_type_ids_list , attention_mask_list , answer_start_positions , answer_end_positions = [] , [] , [] , [] , []
			for i in range(0 , len(paragraph_input_ids) , self.doc_stride):
				token_type_ids = (len(question_input_ids) + 2) * [0] + (self.max_length_of_sequence - len(question_input_ids) - 2) * [1]
				input_ids = [self.tokenizer.cls_token_id] + question_input_ids + [self.tokenizer.sep_token_id] + paragraph_input_ids[i : i + self.max_length_of_paragraph] + [self.tokenizer.sep_token_id]
				attention_mask = len(input_ids) * [1] + max(0 , self.max_length_of_sequence - len(input_ids)) * [0]
				input_ids += max(0 , self.max_length_of_sequence - len(input_ids)) * [self.tokenizer.pad_token_id]
				input_ids_list.append(input_ids)
				token_type_ids_list.append(token_type_ids)
				attention_mask_list.append(attention_mask)
				if self.mode == 'validation':
					if i * self.doc_stride <= self.data[index]['answer']['start'] and self.data[index]['answer']['start'] < (i + 1) * self.doc_stride:
						answer_start_position = self.data[index]['answer']['start'] - i * self.doc_stride + (len(question_input_ids) + 2)
					else:
						answer_start_position = self.max_length_of_sequence + 1
					if i * self.doc_stride <= self.data[index]['answer']['start'] + len(self.data[index]['answer']['text']) - 1 and self.data[index]['answer']['start'] + len(self.data[index]['answer']['text']) - 1 < (i + 1) * self.doc_stride:
						answer_end_position = self.data[index]['answer']['start'] + len(self.data[index]['answer']['text']) - 1 - i * self.doc_stride + (len(question_input_ids) + 2)
					else:
						answer_end_position = self.max_length_of_sequence + 1
					answer_start_positions.append(answer_start_position)
					answer_end_positions.append(answer_end_position)

			if self.mode == 'validation':
				return paragraph , torch.tensor(input_ids_list) , torch.tensor(token_type_ids_list) , torch.tensor(attention_mask_list) , torch.tensor(answer_start_positions) , torch.tensor(answer_end_positions) , self.data[index]['answer']['text']
			else:
				return paragraph , torch.tensor(input_ids_list) , torch.tensor(token_type_ids_list) , torch.tensor(attention_mask_list)

def get_dataloader(context , data , tokenizer , config , task , mode):
	if task == 'MultipleChoice':
		dataset = MultipleChoiceDataset(context , data , tokenizer , config , mode)
	elif task == 'QuestionAnswering':
		dataset = QuestionAnsweringDataset(context , data , tokenizer , config , mode)
	dataloader = DataLoader(dataset , batch_size = (1 if task == 'QuestionAnswering' and mode != 'train' else config['batch_size'] // config['accum_iter']) , shuffle = (mode == 'train') , num_workers = 8)
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

def plot_learning_curve(train_loss_list , validation_loss_list , train_accuracy_list , validation_accuracy_list , task):
	import matplotlib.pyplot as plt

	metric = 'EM' if task == 'QuestionAnswering' else 'Accuracy'

	plt.plot(np.arange(1 , len(train_loss_list) + 1) , train_loss_list , marker = 'o' , label = 'Train')
	plt.plot(np.arange(1 , len(validation_loss_list) + 1) , validation_loss_list , marker = 'o' , label = 'Validation')
	plt.title('Learning Curve')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.xticks(np.arange(1 , len(train_loss_list) + 1))
	plt.legend()
	plt.savefig(f'{task}_loss.png')
	plt.clf()

	plt.plot(np.arange(1 , len(train_accuracy_list) + 1) , train_accuracy_list , marker = 'o' , label = 'Train')
	plt.plot(np.arange(1 , len(validation_accuracy_list) + 1) , validation_accuracy_list , marker = 'o' , label = 'Validation')
	plt.title('Learning Curve')
	plt.xlabel('Epoch')
	plt.ylabel(metric)
	plt.xticks(np.arange(1 , len(train_accuracy_list) + 1))
	plt.legend()
	plt.savefig(f'{task}_{metric}.png')
	plt.clf()