import json

def load_jsonl(file_name):
	data = []
	with open(file_name , 'r') as file:
		for line in file:
			data.append(json.loads(line))
	return data

def save_prediction(test_data , output_file):
	with open(output_file , 'w') as file:
		for i in range(len(test_data)):
			file.write(json.dumps({'id' : test_data[i]['id'] , 'title' : test_data[i]['title']}) + '\n')