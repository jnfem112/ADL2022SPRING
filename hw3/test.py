import torch
import transformers
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM
from accelerate import Accelerator
from utils import my_argparse , get_dataloader
from data import load_jsonl , save_prediction

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

def test(test_data , tokenizer , model , accelerator , device , args):
	test_dataloader = get_dataloader(test_data , None , tokenizer , args , 'test')
	model.to(device)
	model.eval()
	model , test_dataloader = accelerator.prepare(model , test_dataloader)
	prediction = []
	with torch.no_grad():
		for input_ids , attention_mask in test_dataloader:
			input_ids , attention_mask = input_ids.to(device) , attention_mask.to(device)
			output = model.generate(input_ids , max_length = args.max_length_of_target , num_beams = args.num_beams , do_sample = bool(args.do_sample) , top_k = args.top_k , top_p = args.top_p , temperature = args.temperature)
			prediction += tokenizer.batch_decode(output , skip_special_tokens = True)
	for i in range(len(prediction)):
		test_data[i]['title'] = prediction[i]
	return test_data

def main(args):
	accelerator = Accelerator(fp16 = True)
	device = accelerator.device
	print('load data...')
	test_data = load_jsonl(args.test_data)[:10]
	print('test model...')
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_name)
	test_data = test(test_data , tokenizer , model , accelerator , device , args)
	save_prediction(test_data , args.output_file)

if __name__ == '__main__':
	args = my_argparse()
	main(args)