import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
from tqdm import tqdm


def load_model(model_name="kaist-ai/Prometheus-7b-v1.0"):
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
  model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
  return tokenizer, model

def load_dataset(test_file = "benchmark/data/hhh_alignment_eval.json"):
  with open(test_file, 'r') as file:
      data = json.load(file)
  return data

def get_inference(model, data):

  input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
  outputs = model.generate(input_ids, temperature=1.0, top_p=0.9, max_new_tokens=256, repetition_penalty=1.03)
  print(tokenizer.decode(outputs[0]))

def get_res(outputs):
  f_o = outputs.rfind("###Feedback")
  s_o = outputs.rfind("[RESULT]")
  return outputs[f_o:s_o], int(outputs[s_o+9:s_o+10])

  
def make_inferences(model, tokenizer, data, output_path = 'output.csv'):
  res_df = pd.DataFrame()
  fin_accuracy = 0
  n_inferences = 0
  for i in range(len(data)):
    try:
      pos = data[i]['chosen_instruction']
      neg = data[i]['rejected_instruction']

      input_ids = tokenizer(pos, return_tensors="pt").input_ids.to("cuda")
      outputs = model.generate(input_ids, temperature=1.0, top_p=0.9, max_new_tokens=256, repetition_penalty=1.03)
      pos_op = okenizer.decode(outputs[0])

      input_ids = tokenizer(neg, return_tensors="pt").input_ids.to("cuda")
      outputs = model.generate(input_ids, temperature=1.0, top_p=0.9, max_new_tokens=256, repetition_penalty=1.03)
      neg_op = tokenizer.decode(outputs[0])

      pos_feedback, pos_score = get_res(pos_op)
      neg_feedback, neg_score = get_res(neg_op)
      acc = 0
      if pos_score>=neg_score:
          acc = 1

      fin_accuracy+=acc

      res = {}
      res['chosen_instruction'] = pos
      res['rejected_instruction'] = neg
      res['chosen_feedback'] = pos_feedback
      res['chosen_score'] = pos_score
      res['rejected_feedback'] = neg_feedback
      res['rejected_score'] = neg_score
      print(i)
      #print(res)
      res_df = res_df.append(res, ignore_index = True)
      print(res_df)
    except Exception as e:
      print(f"An exception occurred: {e}")
      continue

  res_df.to_csv(output_path)
  tot_acc = fin_accuracy/n_inferences
  return tot_accuracy

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process model, dataset, and output file names.")
    
    # Define command line arguments
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("dataset_path", type=str, help="Name of the dataset")
    parser.add_argument("output_file_name", type=str, help="Name of the output file")

    # Parse the command line arguments
    args = parser.parse_args()

    # Load model
    tokenizer, model = load_model(model_name)
    data = load_dataset(dataset_path)
    acc = make_inferences(model, tokenizer, data, output_path = 'output.csv')
    print('******* Final Accuracy = ', acc)



               
