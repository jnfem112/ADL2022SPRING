import os
import numpy as np
import torch
from transformers import AutoTokenizer , AutoModelForMultipleChoice , AutoModelForQuestionAnswering
from data import load_json

def load_model(checkpoint_path , task):
	config = load_json(os.path.join(checkpoint_path , 'config.json'))
	tokenizer = AutoTokenizer.from_pretrained(config['_name_or_path'])
	if task == 'MultipleChoice':
		model = AutoModelForMultipleChoice.from_pretrained(checkpoint_path)
	elif task == 'QuestionAnswering':
		model = AutoModelForQuestionAnswering.from_pretrained(checkpoint_path)
	return tokenizer , model

def decode_answer(start_logits , end_logits , paragraph , input_ids_list , token_type_ids_list , attention_mask_list , tokenizer , doc_stride , device , mode):
	if mode == 'train':
		answers = []
		for input_ids , token_type_ids , attention_mask , start_logit , end_logit in zip(input_ids_list , token_type_ids_list , attention_mask_list , start_logits , end_logits):
			seq_len = len(input_ids)
			logits = start_logit.unsqueeze(dim = 1) + end_logit.unsqueeze(dim = 0)
			attention_mask[torch.sum(attention_mask).item() - 1] = 0
			condition_1 = (token_type_ids.unsqueeze(dim = 0).bool() * token_type_ids.unsqueeze(dim = 1).bool()).to(device)
			condition_2 = (attention_mask.unsqueeze(dim = 0).bool() * attention_mask.unsqueeze(dim = 1).bool()).to(device)
			condition_3 = torch.triu(torch.ones((seq_len , seq_len))).bool().to(device)
			condition = condition_1 * condition_2 * condition_3
			logits = torch.where(condition , logits , -torch.inf * torch.ones((seq_len , seq_len)).to(device))
			position = torch.argmax(logits).item()
			answer_start_position , answer_end_position = position // seq_len , position % seq_len
			answer = tokenizer.decode(input_ids[answer_start_position : answer_end_position + 1]).replace(' ' , '')
			answers.append(answer)
		return np.array(answers)
	else:
		tokenized_paragraph = tokenizer(paragraph , truncation = False , add_special_tokens = False)
		max_logit , best_answer = -torch.inf , None
		for i , (input_ids , token_type_ids , attention_mask , start_logit , end_logit) in enumerate(zip(input_ids_list , token_type_ids_list , attention_mask_list , start_logits , end_logits)):
			seq_len = len(input_ids)
			logits = start_logit.unsqueeze(dim = 1) + end_logit.unsqueeze(dim = 0)
			attention_mask[torch.sum(attention_mask).item() - 1] = 0
			condition_1 = (token_type_ids.unsqueeze(dim = 0).bool() * token_type_ids.unsqueeze(dim = 1).bool()).to(device)
			condition_2 = (attention_mask.unsqueeze(dim = 0).bool() * attention_mask.unsqueeze(dim = 1).bool()).to(device)
			condition_3 = torch.triu(torch.ones((seq_len , seq_len))).bool().to(device)
			condition = condition_1 * condition_2 * condition_3
			logits = torch.where(condition , logits , -torch.inf * torch.ones((seq_len , seq_len)).to(device))
			position = torch.argmax(logits.view(-1)).item()
			answer_start_position , answer_end_position = position // seq_len , position % seq_len
			logit = logits[answer_start_position][answer_end_position]
			paragraph_start_position = tokenized_paragraph.token_to_chars(i * doc_stride + (answer_start_position - torch.sum(token_type_ids == 0).item())).start
			paragraph_end_position = tokenized_paragraph.token_to_chars(i * doc_stride + (answer_end_position - torch.sum(token_type_ids == 0).item())).end
			answer = paragraph[paragraph_start_position : paragraph_end_position]
			if logit >= max_logit:
				max_logit = logit
				best_answer = answer
		
		if best_answer[0] == '「' and best_answer[-1] != '」':
			best_answer = best_answer[1 : ]
		elif best_answer[0] != '「' and best_answer[-1] == '」':
			best_answer = best_answer[ : -1]
		elif best_answer[0] == '「' and best_answer[-1] == '」':
			best_answer = best_answer[1 : -1]
		elif best_answer[0] == '《' and best_answer[-1] != '》':
			best_answer = best_answer + '》'
		elif best_answer[0] != '《' and best_answer[-1] == '》':
			best_answer = '《' + best_answer

		return best_answer