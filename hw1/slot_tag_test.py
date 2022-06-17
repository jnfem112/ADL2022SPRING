import os
import numpy as np
import torch
from slot_tag_utils import my_argparse , get_dataloader
from slot_tag_data import load_WordVector , load_data , save_prediction
from slot_tag_model import Classifier

def test(test_x , model , device):
	test_dataloader = get_dataloader(test_x , None , 'test')
	model.to(device)
	model.eval()
	test_y = []
	with torch.no_grad():
		for data in test_dataloader:
			data = data.to(device , dtype = torch.long)
			output = model(data)
			_ , index = torch.max(output , dim = 2)
			test_y.append(index.cpu().detach().numpy())
	test_y = np.concatenate(test_y , axis = 0)
	return test_y

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	WordVector = load_WordVector()
	text_id , test_x , _ = load_data(args.test_data , 'test' , WordVector , args.max_length)
	print('test model...')
	model = Classifier(WordVector)
	count = np.zeros((len(test_x) , args.max_length , 10))
	for file_name in os.listdir(args.checkpoint_directory):
		if not file_name.endswith('.pth') or not file_name.startswith('slot'):
			continue
		model.load_state_dict(torch.load(os.path.join(args.checkpoint_directory , file_name) , map_location = device))
		test_y = test(test_x , model , device)
		for i in range(len(test_y)):
			for j in range(args.max_length):
				count[i][j][test_y[i][j]] += 1
	test_y = count.argmax(axis = 2)
	save_prediction(text_id , test_x , test_y , WordVector , args.output_file)

if __name__ == '__main__':
	args = my_argparse()
	main(args)