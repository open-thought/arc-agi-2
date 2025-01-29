import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.hub import cached_file

default_models = [
	"meta-llama/Llama-3.2-1B",
	"meta-llama/Llama-3.2-1B-Instruct",
	# "meta-llama/Llama-3.1-70B-Instruct",
	"Qwen/Qwen2.5-0.5B",
	"Qwen/Qwen2.5-0.5B-Instruct",
	"Qwen/Qwen2.5-1.5B",
	"Qwen/Qwen2.5-1.5B-Instruct",
]

def load_model_and_tokenizer(model: str, local_only: bool = False):
	tokenizer = AutoTokenizer.from_pretrained(
		model,
		token=os.environ["HF_TOKEN"],
		device_map="auto",
		local_files_only=local_only
	)
	model_obj = AutoModelForCausalLM.from_pretrained(
		model,
		token=os.environ["HF_TOKEN"],
		device_map="auto",
		local_files_only=local_only
	)
	model_path = cached_file(model, "tokenizer.json")
	return model_path, tokenizer, model_obj

def download_models(models: list[str]):
	for model in models:
		try:
			# Check if model is already downloaded
			model_path, _, _ = load_model_and_tokenizer(model, local_only=True)
			print(f"{f'Model {model:^40} already exists in cache at:':<80}{model_path}")
		except:
			# If not found in cache, download it
			print(f"{f'Downloading {model:^40}...':<80}")
			model_path, _, _ = load_model_and_tokenizer(model)
			print(f"{f'Saved model {model:^40} to:':<80}{model_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--models", type=str, nargs="+", default=default_models)
	args = parser.parse_args()
	download_models(args.models)
