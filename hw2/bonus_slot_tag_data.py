import json
import pandas as pd

def load_json(file_name):
	with open(file_name , 'r') as file:
		data = json.load(file)
	return data

def save_prediction(test_data , output_file):
	ids , tags = [] , []
	for i in range(len(test_data)):
		ids.append(test_data[i]['id'])
		tags.append(test_data[i]['tags'])
	df = pd.DataFrame({'id' : ids , 'tags' : tags})
	df.to_csv(output_file , index = False)