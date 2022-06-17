import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import transformers
from transformers import AutoConfig , AutoTokenizer , AutoModelForQuestionAnswering , get_linear_schedule_with_warmup
from time import time
from utils import my_argparse , load_config , get_dataloader , print_progress , plot_learning_curve
from data import load_json
from model import decode_answer

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

def train(context , train_data , validation_data , config , tokenizer , model , device , plot):
	train_dataloader = get_dataloader(context , train_data , tokenizer , config , 'QuestionAnswering' , 'train')
	model.to(device)
	optimizer = AdamW(model.parameters() , lr = config['learning_rate'] , weight_decay = config['weight_decay'])
	total_step = config['epoch'] * len(train_dataloader) // config['accum_iter']
	scheduler = get_linear_schedule_with_warmup(optimizer , total_step // 10 , total_step)
	train_loss_list , train_accuracy_list , validation_loss_list , validation_accuracy_list = [] , [] , [] , []
	max_accuracy = 0
	start = time()
	for i in range(config['epoch']):
		model.train()
		count , total_loss = 0 , 0
		start = time()
		for j , (input_ids , token_type_ids , attention_mask , start_positions , end_positions , answer) in enumerate(train_dataloader):
			input_ids , token_type_ids , attention_mask , start_positions , end_positions , answer = input_ids.to(device) , token_type_ids.to(device) , attention_mask.to(device) , start_positions.to(device) , end_positions.to(device) , np.array(answer)
			output = model(input_ids = input_ids , token_type_ids = token_type_ids , attention_mask = attention_mask , start_positions = start_positions , end_positions = end_positions)
			start_logits , end_logits , loss = output.start_logits , output.end_logits , output.loss
			prediction = decode_answer(start_logits , end_logits , None , input_ids , token_type_ids , attention_mask , tokenizer , config['doc_stride'] , device , 'train')
			count += np.sum(prediction == answer)
			total_loss += loss.item()
			loss.backward()
			if (j + 1) % config['accum_iter'] == 0 or j == len(train_dataloader) - 1:
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
			end = time()
			print_progress(i + 1 , config['epoch'] , len(train_data) , config['batch_size'] // config['accum_iter'] , j + 1 , len(train_dataloader) , int(end - start) , total_loss / len(train_data) , count / len(train_data))

		train_loss = total_loss / len(train_data)
		train_accuracy = count / len(train_data)
		train_loss_list.append(train_loss)
		train_accuracy_list.append(train_accuracy)

		validation_loss , validation_accuracy = evaluate(context , validation_data , config , tokenizer , model , device)
		validation_loss_list.append(validation_loss)
		validation_accuracy_list.append(validation_accuracy)
		if validation_accuracy >= max_accuracy:
			print('save model...')
			model.save_pretrained(os.path.join(config['checkpoint_root'] , config['checkpoint_name']))
			max_accuracy = validation_accuracy

	if plot:
		plot_learning_curve(train_loss_list , validation_loss_list , train_accuracy_list , validation_accuracy_list , 'QuestionAnswering')

def evaluate(context , validation_data , config , tokenizer , model , device):
	validation_dataloader = get_dataloader(context , validation_data , tokenizer , config , 'QuestionAnswering' , 'validation')
	model.to(device)
	model.eval()
	count , total_loss = 0 , 0
	start = time()
	with torch.no_grad():
		for paragraph , input_ids , token_type_ids , attention_mask , start_positions , end_positions , answer in validation_dataloader:
			paragraph , input_ids , token_type_ids , attention_mask , start_positions , end_positions , answer = paragraph[0] , input_ids.squeeze(dim = 0).to(device) , token_type_ids.squeeze(dim = 0).to(device) , attention_mask.squeeze(dim = 0).to(device) , start_positions.squeeze(dim = 0).to(device) , end_positions.squeeze(dim = 0).to(device) , answer[0]
			output = model(input_ids = input_ids , token_type_ids = token_type_ids , attention_mask = attention_mask , start_positions = start_positions , end_positions = end_positions)
			start_logits , end_logits , loss = output.start_logits , output.end_logits , output.loss
			prediction = decode_answer(start_logits , end_logits , paragraph , input_ids , token_type_ids , attention_mask , tokenizer , config['doc_stride'] , device , 'validation')
			count += (prediction == answer)
			total_loss += loss.item()
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / len(validation_data) , count / len(validation_data)))
	return total_loss / len(validation_data) , count / len(validation_data)

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	context = load_json(args.context_file)
	train_data = load_json(args.train_data)
	validation_data = load_json(args.validation_data)
	print('train model...')
	_ , config = load_config(args.base_model)
	tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
	if args.pretrained:
		model = AutoModelForQuestionAnswering.from_pretrained(config['model_name'])
	else:
		model_config = AutoConfig.from_pretrained(config['model_name'])
		model = AutoModelForQuestionAnswering.from_config(model_config)
	train(context , train_data , validation_data , config , tokenizer , model , device , args.plot)

if __name__ == '__main__':
	args = my_argparse()
	main(args)