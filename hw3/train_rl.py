import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM , Adafactor
from accelerate import Accelerator
from tw_rouge import get_rouge
import matplotlib.pyplot as plt
from time import time
from utils import my_argparse , get_dataloader , print_progress , plot_learning_curve
from data import load_jsonl

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

def calculate_baseline(train_data , tokenizer , model , accelerator , device , args):
	train_dataloader = get_dataloader(train_data , None , tokenizer , args , 'validation')
	model.to(device)
	model.eval()
	model , train_dataloader = accelerator.prepare(model , train_dataloader)
	baseline = []
	with torch.no_grad():
		for input_ids , attention_mask , labels , titles in train_dataloader:
			input_ids = input_ids.to(device)
			output = model.generate(input_ids , max_length = args.max_length_of_target)
			predictions = tokenizer.batch_decode(output , skip_special_tokens = True)
			for i in range(len(predictions)):
				predictions[i] += '\n'
			scores = get_rouge(predictions , list(titles) , avg = False)
			for score in scores:
				baseline.append(np.mean([score['rouge-1']['f'] , score['rouge-2']['f'] , score['rouge-l']['f']]))
	return baseline

def train(train_data , validation_data , tokenizer , model , accelerator , device , args):
	model.to(device)
	optimizer = Adafactor(model.parameters() , lr = args.learning_rate , weight_decay = args.weight_decay , relative_step = False , scale_parameter = False)
	model , optimizer = accelerator.prepare(model , optimizer)
	criterion = nn.CrossEntropyLoss(reduction = 'none')
	train_loss_list , train_ROUGE_list , validation_loss_list , validation_ROUGE_list = [] , [] , [] , []
	max_ROUGE_1 , max_ROUGE_2 , max_ROUGE_l = 0 , 0 , 0
	for i in range(args.epoch):
		baseline = calculate_baseline(train_data , tokenizer , model , accelerator , device , args)
		train_dataloader = get_dataloader(train_data , baseline , tokenizer , args , 'train')
		train_dataloader = accelerator.prepare(train_dataloader)
		model.train()
		hypothesis , reference = [] , []
		total_loss = 0
		start = time()
		for j , (input_ids , attention_mask , labels , titles , baseline) in enumerate(train_dataloader):
			input_ids , attention_mask , labels , baseline = input_ids.to(device) , attention_mask.to(device) , labels.to(device) , baseline.to(device)

			output = model(input_ids = input_ids , attention_mask = attention_mask , labels = labels)
			ml_loss = output.loss

			output = model.generate(input_ids , num_beams = args.num_beams , do_sample = True , top_k = args.top_k , top_p = args.top_p , temperature = args.temperature , return_dict_in_generate = True , output_scores = True)
			decoder_input_ids = output.sequences
			predictions = tokenizer.batch_decode(decoder_input_ids , skip_special_tokens = True)
			for k in range(len(predictions)):
				predictions[k] += '\n'
			scores = get_rouge(predictions , list(titles) , avg = False)
			reward = torch.FloatTensor([np.mean([score['rouge-1']['f'] , score['rouge-2']['f'] , score['rouge-l']['f']]) for score in scores]).to(device)

			labels = torch.roll(decoder_input_ids , -1 , dims = 1)
			labels[labels == tokenizer.pad_token_id] = -100
			output = model(input_ids , attention_mask = attention_mask , labels = labels)
			batch_size , seq_len , vocab_size = output.logits.shape
			rl_loss = torch.mean(criterion(output.logits.view(-1 , vocab_size) , labels.view(-1)).view(batch_size , seq_len) * (reward - baseline).view(-1 , 1))

			loss = (1 - args.ratio) * ml_loss + args.ratio * rl_loss
			total_loss += loss.item()
			accelerator.backward(loss)
			if (j + 1) % args.accum_iter == 0 or j == len(train_dataloader) - 1:
				optimizer.step()
				optimizer.zero_grad()

			output = model.generate(input_ids , max_length = args.max_length_of_target)
			predictions = tokenizer.batch_decode(output , skip_special_tokens = True)
			for title , prediction in zip(titles , predictions):
				hypothesis.append(prediction + '\n')
				reference.append(title + '\n')

			end = time()
			if j < len(train_dataloader) - 1:
				print_progress(i + 1 , args.epoch , len(train_data) , args.batch_size // args.accum_iter , j + 1 , len(train_dataloader) , int(end - start) , total_loss / len(train_data))
			else:
				score = get_rouge(hypothesis , reference)
				train_loss_list.append(total_loss / len(train_data))
				train_ROUGE_list.append({'ROUGE-1' : score['rouge-1']['f'] , 'ROUGE-2' : score['rouge-2']['f'] , 'ROUGE-L' : score['rouge-l']['f']})
				print_progress(i + 1 , args.epoch , len(train_data) , args.batch_size // args.accum_iter , j + 1 , len(train_dataloader) , int(end - start) , total_loss / len(train_data) , score['rouge-1']['f'] , score['rouge-2']['f'] , score['rouge-l']['f'])

		validation_loss , validation_ROUGE_1 , validation_ROUGE_2 , validation_ROUGE_l = evaluate(validation_data , tokenizer , model , accelerator , device , args)
		validation_loss_list.append(validation_loss)
		validation_ROUGE_list.append({'ROUGE-1' : validation_ROUGE_1 , 'ROUGE-2' : validation_ROUGE_2 , 'ROUGE-L' : validation_ROUGE_l})
		if validation_ROUGE_1 >= max_ROUGE_1 and validation_ROUGE_2 >= max_ROUGE_2 and validation_ROUGE_l >= max_ROUGE_l:
			print('save model...')
			model.save_pretrained(args.checkpoint_name)
			max_ROUGE_1 , max_ROUGE_2 , max_ROUGE_l = validation_ROUGE_1 , validation_ROUGE_2 , validation_ROUGE_l
	
	if args.plot:
		plot_learning_curve(train_loss_list , validation_loss_list , train_ROUGE_list , validation_ROUGE_list)

def evaluate(validation_data , tokenizer , model , accelerator , device , args):
	validation_dataloader = get_dataloader(validation_data , None , tokenizer , args , 'validation')
	model.to(device)
	model.eval()
	model , validation_dataloader = accelerator.prepare(model , validation_dataloader)
	total_loss = 0
	hypothesis , reference = [] , []
	start = time()
	with torch.no_grad():
		for input_ids , attention_mask , labels , titles in validation_dataloader:
			input_ids , attention_mask , labels = input_ids.to(device) , attention_mask.to(device) , labels.to(device)
			output = model(input_ids = input_ids , attention_mask = attention_mask , labels = labels)
			total_loss += output.loss.item()

			output = model.generate(input_ids , max_length = args.max_length_of_target)
			predictions = tokenizer.batch_decode(output , skip_special_tokens = True)
			for title , prediction in zip(titles , predictions):
				hypothesis.append(prediction + '\n')
				reference.append(title + '\n')
	score = get_rouge(hypothesis , reference)
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation ROUGE-1 : {:.1f} , validation ROUGE-2 : {:.1f} , validation ROUGE-L : {:.1f}'.format(int(end - start) , total_loss / len(validation_data) , 100 * score['rouge-1']['f'] , 100 * score['rouge-2']['f'] , 100 * score['rouge-l']['f']))
	return total_loss / len(validation_data) , score['rouge-1']['f'] , score['rouge-2']['f'] , score['rouge-l']['f']

def main(args):
	accelerator = Accelerator(fp16 = True)
	device = accelerator.device
	print('load data...')
	train_data = load_jsonl(args.train_data)
	validation_data = load_jsonl(args.validation_data)
	print('train model...')
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
	train(train_data , validation_data , tokenizer , model , accelerator , device , args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)