<h1 align="center">Prometheus</h1>

This code is derived from the official github repository for [Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491).


Prometheus is an evaluator LM that is open-source, offers reproducible evaluation, and inexpensive to use. Specifically designed for fine-grained evaluation on a customized score rubric, Prometheus is a good alternative for human evaluation and GPT-4 evaluation.


Citation:
```
@article{kim2023prometheus,
  title={Prometheus: Inducing Fine-grained Evaluation Capability in Language Models},
  author={Kim, Seungone and Shin, Jamin and Cho, Yejin and Jang, Joel and Longpre, Shayne and Lee, Hwaran and Yun, Sangdoo and Shin, Seongjin and Kim, Sungdong and Thorne, James and others},
  journal={arXiv preprint arXiv:2310.08491},
  year={2023}
}
```
### Setup

Install dependencies

```
pip install -r requirements.txt
```

Log into Hugging Face. To run inferences, and our models, please ensure you have access to Llama2 weights on HuggingFace. If not, follow the instructions present here: https://ai.meta.com/resources/models-and-libraries/llama-downloads/
```
huggingface-cli login
```
   
### Input and Output Format of Prometheus

Prometheus is trained and inferenced using the following input prompt format. Note that you could fill in the instruction, response, reference answer, and score rubrics with your own data.

```
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score 1: {orig_score1_description}
Score 2: {orig_score2_description}
Score 3: {orig_score3_description}
Score 4: {orig_score4_description}
Score 5: {orig_score5_description}

###Feedback: 
```

Also, we use the following output format.

```
{orig_feedback}
[RESULT] {orig_score}
```

During inference, we parse the score by splitting the number that is generated next to the \[RESULT\] phrase.


### Training an Evaluator LM



### Inference with an Evaluator LM
1) To run scoring for a dataset with one single instruction, run the following command - 

  python evaluation/fin_single_run.py --model_name "kaist-ai/Prometheus-7b-v1.0" --dataset_path "evaluation/benchmark/data/feedback_collection_test.json" --output_file_name ""
  
  The applicable datasets are - feedback_collection_test.json, vicuna_eval.json

2) To run scoring for a dataset with 2 instructions, run the following command - 

  python evaluation/fin_eval_run.py --model_name "kaist-ai/Prometheus-7b-v1.0" --dataset_path "evaluation/benchmark/data/hhh_alignment_eval.json" --output_file_name ""

  The applicable datasets are - hhh_alignment_eval.json

