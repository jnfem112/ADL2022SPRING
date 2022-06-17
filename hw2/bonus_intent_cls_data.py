import json
import pandas as pd

def load_json(file_name):
	with open(file_name , 'r') as file:
		data = json.load(file)
	return data

def save_prediction(test_data , output_file):
	ids , intents = [] , []
	for i in range(len(test_data)):
		ids.append(test_data[i]['id'])
		intents.append(test_data[i]['intent'])
	df = pd.DataFrame({'id' : ids , 'intent' : intents})
	df.to_csv(output_file , index = False)