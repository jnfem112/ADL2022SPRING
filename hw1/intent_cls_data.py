import json
import pandas as pd
import numpy as np
import gensim
import gensim.downloader
import spacy

def load_WordVector():
	WordVector = gensim.downloader.load('glove-wiki-gigaword-200')
	WordVector.add_vector('<PAD>' , np.zeros((200)))
	WordVector.add_vector('<UNK>' , np.zeros((200)))
	return WordVector

def load_data(file_name , mode , WordVector , max_length):
	NLP = spacy.load('en_core_web_lg')
	with open('intent_cls_label.json' , 'r') as file:
		data = json.load(file)
	intent2label = {data[i]['intent'] : data[i]['label'] for i in range(len(data))}
	text_id , text , label = [] , [] , []

	with open(file_name , 'r') as file:
		data = json.load(file)
		for i in range(len(data)):
			text_id.append(data[i]['id'])
			text.append([WordVector.get_index(token.lemma_) if token.lemma_ in WordVector else WordVector.get_index('<UNK>') for token in NLP(data[i]['text'])])
			if mode != 'test':
				label.append(intent2label[data[i]['intent']])
	
	for i in range(len(text)):
		text[i] = text[i][ : min(len(text[i]) , max_length)]
		text[i] += [WordVector.get_index('<PAD>') for _ in range(max_length - len(text[i]))]

	return text_id , text , label

def save_prediction(text_id , prediction , output_file):
	with open('intent_cls_label.json' , 'r') as file:
		data = json.load(file)
	label2intent = {data[i]['label'] : data[i]['intent'] for i in range(len(data))}
	prediction = [label2intent[prediction[i]] for i in range(len(prediction))]
	df = pd.DataFrame({'id' : text_id , 'intent' : prediction})
	df.to_csv(output_file , index = False)