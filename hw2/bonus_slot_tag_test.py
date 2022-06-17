import os
import numpy as np
import torch
import transformers
from bonus_slot_tag_utils import my_argparse , load_config , get_dataloader
from bonus_slot_tag_data import load_json , save_prediction
from bonus_slot_tag_model import load_model , decode_tags

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

def test(test_data , config , label2tag , tokenizer , model , device):
	test_dataloader = get_dataloader(test_data , tokenizer , None , config , 'test')
	model.to(device)
	model.eval()
	prediction = []
	with torch.no_grad():
		for input_ids , token_type_ids , attention_mask , mask in test_dataloader:
			input_ids , token_type_ids , attention_mask , mask = input_ids.to(device) , token_type_ids.to(device) , attention_mask.to(device) , mask.to(device)
			output = model(input_ids = input_ids , token_type_ids = token_type_ids , attention_mask = attention_mask)
			logits = output.logits
			tags = decode_tags(logits , mask , label2tag)
			prediction.append(tags)
	prediction = np.concatenate(prediction , axis = 0)
	for i in range(len(prediction)):
		test_data[i]['tags'] = prediction[i]
	return test_data

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	test_data = load_json(args.test_data)
	slot_tag_label = load_json('slot_tag_label.json')
	label2tag = {slot_tag_label[i]['label'] : slot_tag_label[i]['tag'] for i in range(len(slot_tag_label))}
	print('test model...')
	config = load_config()
	tokenizer , model = load_model(os.path.join(config['checkpoint_root'] , config['checkpoint_name']))
	test_data = test(test_data , config , label2tag , tokenizer , model , device)
	save_prediction(test_data , args.output_file)

if __name__ == '__main__':
	args = my_argparse()
	main(args)