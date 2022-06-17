import json
import pandas as pd
import numpy as np
import gensim
import gensim.downloader

def load_WordVector():
	WordVector = gensim.downloader.load('glove-wiki-gigaword-200')
	WordVector.add_vector('<PAD>' , np.zeros((200)))
	WordVector.add_vector('<UNK>' , np.zeros((200)))
	return WordVector

def load_data(file_name , mode , WordVector , max_length):
	with open('slot_tag_label.json' , 'r') as file:
		data = json.load(file)
	tag2label = {data[i]['tag'] : data[i]['label'] for i in range(len(data))}
	text_id , tokens , labels , masks = [] , [] , [] , []

	with open(file_name , 'r') as file:
		data = json.load(file)
		for i in range(len(data)):
			text_id.append(data[i]['id'])
			tokens.append([WordVector.get_index(token) if token in WordVector else WordVector.get_index('<UNK>') for token in data[i]['tokens']])
			if mode != 'test':
				labels.append([tag2label[tag] for tag in data[i]['tags']])
	
	for i in range(len(tokens)):
		masks.append(len(tokens[i]) * [1] + (max_length - len(tokens[i])) * [0])
		tokens[i] += [WordVector.get_index('<PAD>') for _ in range(max_length - len(tokens[i]))]
		if mode != 'test':
			labels[i] += [-1 for _ in range(max_length - len(labels[i]))]

	return text_id , tokens , labels , masks

def save_prediction(text_id , test_x , test_y , WordVector , output_file):
	with open('slot_tag_label.json' , 'r') as file:
		data = json.load(file)
	label2tag = {data[i]['label'] : data[i]['tag'] for i in range(len(data))}
	tags = []
	for i in range(len(test_y)):
		tag = []
		for j in range(len(test_y[i])):
			if test_x[i][j] == WordVector.get_index('<PAD>'):
				break
			tag.append(label2tag[test_y[i][j]])
		tags.append(' '.join(tag))
	df = pd.DataFrame({'id' : text_id , 'tags' : tags})
	df.to_csv(output_file , index = False)