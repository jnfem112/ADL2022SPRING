import os
import numpy as np
import torch
from bonus_utils import my_argparse , get_dataloader
from bonus_data import load_WordVector , load_data , save_prediction
from bonus_model import Classifier

def test(test_x , test_mask , model , device):
	test_dataloader = get_dataloader(test_x , None , test_mask , 'test')
	model.to(device)
	model.eval()
	test_y = []
	with torch.no_grad():
		for data , mask in test_dataloader:
			data , mask = data.to(device , dtype = torch.long) , mask.to(device , dtype = torch.uint8)
			prediction = model.inference(data , mask)
			test_y += prediction
	return test_y

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	WordVector = load_WordVector()
	text_id , test_x , _ , test_mask = load_data(args.test_data , 'test' , WordVector , args.max_length)
	print('test model...')
	model = Classifier(WordVector)
	count = np.zeros((len(test_x) , args.max_length , 10))
	for file_name in os.listdir(args.checkpoint_directory):
		if not file_name.endswith('.pth') or not file_name.startswith('slot') or 'bonus' not in file_name:
			continue
		model.load_state_dict(torch.load(os.path.join(args.checkpoint_directory , file_name) , map_location = device))
		test_y = test(test_x , test_mask , model , device)
		for i in range(len(test_y)):
			for j in range(len(test_y[i])):
				count[i][j][test_y[i][j]] += 1
	test_y = count.argmax(axis = 2)
	save_prediction(text_id , test_x , test_y , WordVector , args.output_file)

if __name__ == '__main__':
	args = my_argparse()
	main(args)