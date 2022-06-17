import os
import torch
import transformers
from time import time
from utils import my_argparse , load_config , get_dataloader
from data import load_json , save_prediction
from model import load_model , decode_answer

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

def test(context , test_data , config_1 , config_2 , tokenizer_list_1 , tokenizer_list_2 , model_list_1 , model_list_2 , device):
	print('    multiple choice...')
	logits_list = []
	for i , (tokenizer_1 , model_1) in enumerate(zip(tokenizer_list_1 , model_list_1)):
		test_dataloader_1 = get_dataloader(context , test_data , tokenizer_1 , config_1 , 'MultipleChoice' , 'test')
		model_1.to(device)
		model_1.eval()
		logits = []
		start = time()
		with torch.no_grad():
			for j , (input_ids , token_type_ids , attention_mask) in enumerate(test_dataloader_1):
				input_ids , token_type_ids , attention_mask = input_ids.to(device) , token_type_ids.to(device) , attention_mask.to(device)
				output = model_1(input_ids = input_ids , token_type_ids = token_type_ids , attention_mask = attention_mask)
				logits.append(output.logits)
				end = time()
				print(f'\r        model {i + 1}/{len(model_list_1)} , batch {j + 1}/{len(test_dataloader_1)} ({int(end - start)}s)' , end = '' if j < len(test_dataloader_1) - 1 else '\n')
		logits = torch.cat(logits , dim = 0)
		logits_list.append(logits)
	logits_list = torch.mean(torch.stack(logits_list , dim = 0) , dim = 0)
	_ , index = torch.max(logits_list , dim = 1)
	choices = index.cpu().detach().numpy()
	for i in range(len(test_data)):
		test_data[i]['relevant'] = test_data[i]['paragraphs'][choices[i]]

	print('    question answering...')
	start_logits_list , end_logits_list = [] , []
	for i , (tokenizer_2 , model_2) in enumerate(zip(tokenizer_list_2 , model_list_2)):
		test_dataloader_2 = get_dataloader(context , test_data , tokenizer_2 , config_2 , 'QuestionAnswering' , 'test')
		model_2.to(device)
		model_2.eval()
		start_logits , end_logits = [] , []
		start = time()
		with torch.no_grad():
			for j , (paragraph , input_ids , token_type_ids , attention_mask) in enumerate(test_dataloader_2):
				paragraph , input_ids , token_type_ids , attention_mask = paragraph[0] , input_ids.squeeze(dim = 0).to(device) , token_type_ids.squeeze(dim = 0).to(device) , attention_mask.squeeze(dim = 0).to(device)
				output = model_2(input_ids = input_ids , token_type_ids = token_type_ids , attention_mask = attention_mask)
				start_logits.append(output.start_logits)
				end_logits.append(output.end_logits)
				end = time()
				print(f'\r        model {i + 1}/{len(model_list_2)} , batch {j + 1}/{len(test_dataloader_2)} ({int(end - start)}s)' , end = '' if j < len(test_dataloader_2) - 1 else '\n')
		start_logits_list.append(start_logits)
		end_logits_list.append(end_logits)
	start_logits_list = [torch.mean(torch.stack([start_logits_list[j][i] for j in range(len(model_list_2))] , dim = 0) , dim = 0) for i in range(len(test_dataloader_2))]
	end_logits_list = [torch.mean(torch.stack([end_logits_list[j][i] for j in range(len(model_list_2))] , dim = 0) , dim = 0) for i in range(len(test_dataloader_2))]

	print('    decode answer...')
	start = time()
	for i , ((paragraph , input_ids , token_type_ids , attention_mask) , start_logits , end_logits) in enumerate(zip(test_dataloader_2 , start_logits_list , end_logits_list)):
		paragraph , input_ids , token_type_ids , attention_mask = paragraph[0] , input_ids.squeeze(dim = 0) , token_type_ids.squeeze(dim = 0) , attention_mask.squeeze(dim = 0)
		answer = decode_answer(start_logits , end_logits , paragraph , input_ids , token_type_ids , attention_mask , tokenizer_2 , config_2['doc_stride'] , device , 'test')
		test_data[i]['answer'] = answer
		end = time()
		print(f'\r        batch {i + 1}/{len(test_dataloader_2)} ({int(end - start)}s)' , end = '' if i < len(test_dataloader_2) - 1 else '\n')

	return test_data

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	context = load_json(args.context_file)
	test_data = load_json(args.test_data)
	print('test model...')
	config_1 , config_2 = load_config(args.base_model)
	tokenizer_list_1 , model_list_1 = [] , []
	for file_name in os.listdir(config_1['checkpoint_root']):
		if 'MultipleChoice' in file_name and os.path.isdir(os.path.join(config_1['checkpoint_root'] , file_name)):
			tokenizer , model = load_model(os.path.join(config_1['checkpoint_root'] , file_name) , 'MultipleChoice')
			model_list_1.append(model)
			tokenizer_list_1.append(tokenizer)
	tokenizer_list_2 , model_list_2 = [] , []
	for file_name in os.listdir(config_2['checkpoint_root']):
		if 'QuestionAnswering' in file_name and os.path.isdir(os.path.join(config_2['checkpoint_root'] , file_name)):
			tokenizer , model = load_model(os.path.join(config_2['checkpoint_root'] , file_name) , 'QuestionAnswering')
			model_list_2.append(model)
			tokenizer_list_2.append(tokenizer)
	test_data = test(context , test_data , config_1 , config_2 , tokenizer_list_1 , tokenizer_list_2 , model_list_1 , model_list_2 , device)
	save_prediction(test_data , args.output_file)

if __name__ == '__main__':
	args = my_argparse()
	main(args)