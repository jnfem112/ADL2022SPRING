import os
import torch
import torch.nn as nn
from torch.optim import Adam
from time import time
from bonus_utils import my_argparse , get_dataloader , print_progress
from bonus_data import load_WordVector , load_data
from bonus_model import Classifier

def train(train_x , train_y , train_mask , validation_x , validation_y , validation_mask , WordVector , model , device , args):
	train_dataloader = get_dataloader(train_x , train_y , train_mask , 'train' , args.batch_size)
	model.to(device)
	optimizer = Adam(model.parameters() , lr = args.learning_rate)
	max_accuracy = 0
	for i in range(args.epoch):
		model.train()
		count , total_loss = 0 , 0
		start = time()
		for j , (data , label , mask) in enumerate(train_dataloader):
			data , label , mask = data.to(device , dtype = torch.long) , label.to(device , dtype = torch.long) , mask.to(device , dtype = torch.uint8)
			optimizer.zero_grad()
			output = model(data , label , mask)
			prediction = model.inference(data , mask)
			for k in range(len(prediction)):
				correct = True
				for l in range(len(prediction[k])):
					if prediction[k][l] != label[k][l].item():
						correct = False
				if correct:
					count += 1
			loss = -output
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , args.epoch , len(train_x) , args.batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss / len(train_x) , count / len(train_x))

		accuracy = evaluate(validation_x , validation_y , validation_mask , WordVector , model , device)
		if accuracy >= max_accuracy:
			print('save model...')
			torch.save(model.state_dict() , os.path.join(args.checkpoint_directory , args.checkpoint))
			max_accuracy = accuracy

	return model

def evaluate(validation_x , validation_y , validation_mask , WordVector , model , device):
	validation_dataloader = get_dataloader(validation_x , validation_y , validation_mask , 'validation')
	model.to(device)
	model.eval()
	count , total_loss = 0 , 0
	start = time()
	with torch.no_grad():
		for data , label , mask in validation_dataloader:
			data , label , mask = data.to(device , dtype = torch.long) , label.to(device , dtype = torch.long) , mask.to(device , dtype = torch.uint8)
			output = model(data , label , mask)
			prediction = model.inference(data , mask)
			for i in range(len(prediction)):
				correct = True
				for j in range(len(prediction[i])):
					if prediction[i][j] != label[i][j].item():
						correct = False
				if correct:
					count += 1
			loss = -output
			total_loss += loss.item()
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / len(validation_x) , count / len(validation_x)))
	return count / len(validation_x)

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	WordVector = load_WordVector()
	_ , train_x , train_y , train_mask = load_data(args.train_data , 'train' , WordVector , args.max_length)
	_ , validation_x , validation_y , validation_mask = load_data(args.validation_data , 'validation' , WordVector , args.max_length)
	print('train model...')
	model = Classifier(WordVector)
	train(train_x , train_y , train_mask , validation_x , validation_y , validation_mask , WordVector , model , device , args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)