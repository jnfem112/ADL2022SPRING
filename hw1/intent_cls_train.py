import os
import torch
import torch.nn as nn
from torch.optim import Adam
from time import time
from intent_cls_utils import my_argparse , get_dataloader , print_progress
from intent_cls_data import load_WordVector , load_data
from intent_cls_model import Classifier

def train(train_x , train_y , validation_x , validation_y , model , device , args):
	train_dataloader = get_dataloader(train_x , train_y , 'train' , args.batch_size)
	model.to(device)
	optimizer = Adam(model.parameters() , lr = args.learning_rate)
	criterion = nn.CrossEntropyLoss()
	max_accuracy = 0
	for i in range(args.epoch):
		model.train()
		count , total_loss = 0 , 0
		start = time()
		for j , (data , label) in enumerate(train_dataloader):
			data , label = data.to(device , dtype = torch.long) , label.to(device , dtype = torch.long)
			optimizer.zero_grad()
			output = model(data)
			_ , index = torch.max(output , dim = 1)
			count += torch.sum(label == index).item()
			loss = criterion(output , label)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , args.epoch , len(train_x) , args.batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss / len(train_x) , count / len(train_x))

		accuracy = evaluate(validation_x , validation_y , model , device)
		if accuracy >= max_accuracy:
			print('save model...')
			torch.save(model.state_dict() , os.path.join(args.checkpoint_directory , args.checkpoint))
			max_accuracy = accuracy

	return model

def evaluate(validation_x , validation_y , model , device):
	validation_dataloader = get_dataloader(validation_x , validation_y , 'validation')
	model.to(device)
	model.eval()
	criterion = nn.CrossEntropyLoss()
	count , total_loss = 0 , 0
	start = time()
	with torch.no_grad():
		for data , label in validation_dataloader:
			data , label = data.to(device , dtype = torch.long) , label.to(device , dtype = torch.long)
			output = model(data)
			_ , index = torch.max(output , dim = 1)
			count += torch.sum(label == index).item()
			loss = criterion(output , label)
			total_loss += loss.item()
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / len(validation_x) , count / len(validation_x)))
	return count / len(validation_x)

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	WordVector = load_WordVector()
	_ , train_x , train_y = load_data(args.train_data , 'train' , WordVector , args.max_length)
	_ , validation_x , validation_y = load_data(args.validation_data , 'validation' , WordVector , args.max_length)
	print('train model...')
	model = Classifier(WordVector)
	train(train_x , train_y , validation_x , validation_y , model , device , args)

if __name__ == '__main__':
    args = my_argparse()
    main(args)