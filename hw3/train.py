import torch
import transformers
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM , Adafactor
from accelerate import Accelerator
from tw_rouge import get_rouge
from time import time
from utils import my_argparse , get_dataloader , print_progress , plot_learning_curve
from data import load_jsonl

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

def train(train_data , validation_data , tokenizer , model , accelerator , device , args):
	train_dataloader = get_dataloader(train_data , None , tokenizer , args , 'train')
	model.to(device)
	optimizer = Adafactor(model.parameters() , lr = args.learning_rate , weight_decay = args.weight_decay , relative_step = False , scale_parameter = False)
	train_dataloader , model , optimizer = accelerator.prepare(train_dataloader , model , optimizer)
	train_loss_list , train_ROUGE_list , validation_loss_list , validation_ROUGE_list = [] , [] , [] , []
	max_ROUGE_1 , max_ROUGE_2 , max_ROUGE_l = 0 , 0 , 0
	for i in range(args.epoch):
		model.train()
		total_loss = 0
		hypothesis , reference = [] , []
		start = time()
		for j , (input_ids , attention_mask , labels , titles) in enumerate(train_dataloader):
			input_ids , attention_mask , labels = input_ids.to(device) , attention_mask.to(device) , labels.to(device)
			output = model(input_ids = input_ids , attention_mask = attention_mask , labels = labels)
			total_loss += output.loss.item()
			accelerator.backward(output.loss)
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