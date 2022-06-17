import os
import torch
from transformers import AutoTokenizer , AutoModelForTokenClassification
from bonus_slot_tag_data import load_json

def load_model(checkpoint_path):
	config = load_json(os.path.join(checkpoint_path , 'config.json'))
	tokenizer = AutoTokenizer.from_pretrained(config['_name_or_path'])
	model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
	return tokenizer , model

def decode_tags(logits , mask , label2tag):
	_ , index = torch.max(logits , dim = 2)
	tags_list = []
	for i in range(len(mask)):
		tags = []
		for j in range(len(mask[i])):
			if mask[i][j].item():
				tags.append(label2tag[index[i][j].item()])
		tags = ' '.join(tags)
		tags_list.append(tags)
	return tags_list