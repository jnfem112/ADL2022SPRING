import os
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from bonus_intent_cls_data import load_json

def load_model(checkpoint_path):
	config = load_json(os.path.join(checkpoint_path , 'config.json'))
	tokenizer = AutoTokenizer.from_pretrained(config['_name_or_path'])
	model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
	return tokenizer , model