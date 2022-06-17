import os
import numpy as np
import torch
import transformers
from bonus_intent_cls_utils import my_argparse , load_config , get_dataloader
from bonus_intent_cls_data import load_json , save_prediction
from bonus_intent_cls_model import load_model

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

def test(test_data , config , label2intent , tokenizer , model , device):
	test_dataloader = get_dataloader(test_data , tokenizer , None , config , 'test')
	model.to(device)
	model.eval()
	prediction = []
	with torch.no_grad():
		for input_ids , token_type_ids , attention_mask in test_dataloader:
			input_ids , token_type_ids , attention_mask = input_ids.to(device) , token_type_ids.to(device) , attention_mask.to(device)
			output = model(input_ids = input_ids , token_type_ids = token_type_ids , attention_mask = attention_mask)
			_ , index = torch.max(output.logits , dim = 1)
			prediction.append(index.cpu().detach().numpy())
	prediction = np.concatenate(prediction , axis = 0)
	for i in range(len(prediction)):
		test_data[i]['intent'] = label2intent[prediction[i]]
	return test_data

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	test_data = load_json(args.test_data)
	intent_cls_label = load_json('intent_cls_label.json')
	label2intent = {intent_cls_label[i]['label'] : intent_cls_label[i]['intent'] for i in range(len(intent_cls_label))}
	print('test model...')
	config = load_config()
	tokenizer , model = load_model(os.path.join(config['checkpoint_root'] , config['checkpoint_name']))
	test_data = test(test_data , config , label2intent , tokenizer , model , device)
	save_prediction(test_data , args.output_file)

if __name__ == '__main__':
	args = my_argparse()
	main(args)