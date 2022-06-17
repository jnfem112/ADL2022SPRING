import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import transformers
from transformers import AutoConfig , AutoTokenizer , AutoModelForTokenClassification , get_linear_schedule_with_warmup
from time import time
from bonus_slot_tag_utils import my_argparse , load_config , get_dataloader , print_progress
from bonus_slot_tag_data import load_json

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

def train(train_data , validation_data , config , tag2label , tokenizer , model , device):
	train_dataloader = get_dataloader(train_data , tokenizer , tag2label , config , 'train')
	model.to(device)
	optimizer = AdamW(model.parameters() , lr = config['learning_rate'] , weight_decay = config['weight_decay'])
	total_step = config['epoch'] * len(train_dataloader) // config['accum_iter']
	scheduler = get_linear_schedule_with_warmup(optimizer , total_step // 10 , total_step)
	max_accuracy = 0
	for i in range(config['epoch']):
		model.train()
		count , total_loss = 0 , 0
		start = time()
		for j , (input_ids , token_type_ids , attention_mask , mask , labels) in enumerate(train_dataloader):
			input_ids , token_type_ids , attention_mask , labels = input_ids.to(device) , token_type_ids.to(device) , attention_mask.to(device) , labels.to(device)
			output = model(input_ids = input_ids , token_type_ids = token_type_ids , attention_mask = attention_mask , labels = labels)
			logits , loss = output.logits , output.loss
			_ , index = torch.max(logits , dim = 2)
			for k in range(len(mask)):
				correct = True
				for l in range(len(mask[k])):
					if mask[k][l].item() and index[k][l].item() != labels[k][l].item():
						correct = False
				if correct:
					count += 1
			total_loss += loss.item()
			loss.backward()
			if (j + 1) % config['accum_iter'] == 0 or j == len(train_dataloader) - 1:
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
			end = time()
			print_progress(i + 1 , config['epoch'] , len(train_data) , config['batch_size'] // config['accum_iter'] , j + 1 , len(train_dataloader) , int(end - start) , total_loss / len(train_data) , count / len(train_data))

		accuracy = evaluate(validation_data , config , tag2label , tokenizer , model , device)
		if accuracy >= max_accuracy:
			print('save model...')
			model.save_pretrained(os.path.join(config['checkpoint_root'] , config['checkpoint_name']))
			max_accuracy = accuracy

def evaluate(validation_data , config , tag2label , tokenizer , model , device):
	validation_dataloader = get_dataloader(validation_data , tokenizer , tag2label , config , 'validation')
	model.to(device)
	model.eval()
	count , total_loss = 0 , 0
	start = time()
	with torch.no_grad():
		for input_ids , token_type_ids , attention_mask , mask , labels in validation_dataloader:
			input_ids , token_type_ids , attention_mask , labels = input_ids.to(device) , token_type_ids.to(device) , attention_mask.to(device) , labels.to(device)
			output = model(input_ids = input_ids , token_type_ids = token_type_ids , attention_mask = attention_mask , labels = labels)
			logits , loss = output.logits , output.loss
			_ , index = torch.max(logits , dim = 2)
			for i in range(len(mask)):
				correct = True
				for j in range(len(mask[i])):
					if mask[i][j].item() and index[i][j].item() != labels[i][j].item():
						correct = False
				if correct:
					count += 1
			total_loss += loss.item()
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / len(validation_data) , count / len(validation_data)))
	return count / len(validation_data)

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	train_data = load_json(args.train_data)
	validation_data = load_json(args.validation_data)
	slot_tag_label = load_json('slot_tag_label.json')
	tag2label = {slot_tag_label[i]['tag'] : slot_tag_label[i]['label'] for i in range(len(slot_tag_label))}
	print('train model...')
	config = load_config()
	tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
	model = AutoModelForTokenClassification.from_pretrained(config['model_name'] , num_labels = len(tag2label))
	train(train_data , validation_data , config , tag2label , tokenizer , model , device)

if __name__ == '__main__':
	args = my_argparse()
	main(args)