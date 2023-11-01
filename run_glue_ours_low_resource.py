# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn import CrossEntropyLoss, MSELoss
from utils import load_corpus, ET_Sentiment2_Textonly_Dataset, count_parameters, calculate_mean_std
from glue_model import Eyettention, PLM_AS, Joint_Gaze_LM
from scipy.special import softmax
from sklearn import metrics
import copy

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils import model_zoo
from tqdm.auto import tqdm

import transformers
from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	DataCollatorWithPadding,
	PretrainedConfig,
	SchedulerType,
	default_data_collator,
	get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")
logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
	"cola": ("sentence", None),
	"mnli": ("premise", "hypothesis"),
	"mrpc": ("sentence1", "sentence2"),
	"qnli": ("question", "sentence"),
	"qqp": ("question1", "question2"),
	"rte": ("sentence1", "sentence2"),
	"sst2": ("sentence", None),
	"stsb": ("sentence1", "sentence2"),
	"wnli": ("sentence1", "sentence2"),
}


task_to_metric = {'rte': 'accuracy',
					'mrpc': 'f1',
					'stsb': 'spearmanr',
					'sst2': 'accuracy',
					'cola': 'matthews_correlation',
					'qqp': 'f1',
					'mnli': 'accuracy',
					'qnli': 'accuracy'}

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def remove_punctuation_split(text):
	#The input text has been pre-processed with punctuation splits, so we remove the space before the punctuation and return it to its original form
	txt = text.replace(' ,', ',').replace(' .', '.').replace(' "', '"').replace('“ ', '“').replace(' ”', '”').replace('[ ', '[').replace(' ]', ']').replace('< ', '<').replace(' >', '>').replace(' !', '!').replace(' ?', '?').replace(' ;', ';').replace(' :', ':').replace(" '" , "'").replace(" ’" , "'").replace("‘ " , "‘").replace('$ ' , '$').replace(' %' , '%').replace(' )' , ')').replace('( ' , '(').replace(' _' , '_').replace(' -' , '-').replace(' –' , '–').replace('C + +' , 'C++').replace('`` ', '``').replace('` ', '`')
	if txt[:2] == '" ':
		txt = txt[0] + txt[2:]
	return txt

def compute_word_length(txt):
	txt_word_len = [len(t) for t in txt[1:-1]]
	#pad nan for CLS and SEP tokens
	txt_word_len = [np.nan] + txt_word_len + [np.nan]
	#length of a punctuation is 0, plus an epsilon to avoid division output inf
	arr = np.array(txt_word_len).astype('float64')
	arr[arr==0] = 1/(0+0.5)
	arr[arr!=0] = 1/(arr[arr!=0])
	return arr

def pad_seq(seqs, max_len, dtype=np.compat.long, fill_value=np.nan):
	padded = np.full((len(seqs), max_len), fill_value=fill_value, dtype=dtype)
	for i, seq in enumerate(seqs):
		padded[i, :len(seq)] = seq
	return padded

def parse_args():
	parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
	parser.add_argument(
		"--gpu",
		type=int,
		default=0,
		help="The GPU index that we want to use.",
	)

	parser.add_argument(
		"--task_name",
		type=str,
		default='rte',
		help="The name of the glue task to train on.",
		choices=list(task_to_keys.keys()),
	)

	parser.add_argument(
		"--max_length",
		type=int,
		default=512,
		help=(
			"The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
			" sequences shorter will be padded if `--pad_to_max_length` is passed."
		),
	)

	parser.add_argument(
		"--model_name_or_path",
		type=str,
		default = 'bert-base-cased',
		help = "Path to pretrained model or model identifier from huggingface.co/models."
		)

	parser.add_argument(
		"--batch_size",
		type=int,
		default=32,
		help="Batch size (per device) for the training dataloader.",
	)

	parser.add_argument(
		"--max_train_samples",
		type=int,
		default=1000,
		help="truncate the number of training examples to this value if set.",
	)

	parser.add_argument(
		"--tau",
		type=float,
		default=0.5,
		help="Initial learning rate (after the potential warmup period) to use.",
	)

	parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
	parser.add_argument("--output_dir", type=str, default='glue_low_resource_results/', help="Where to store the final model.")
	parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
	parser.add_argument("--data_seed", type=int, default=111, help="A seed for reproducible training.")
	parser.add_argument("--earlystop_patience", type=int, default=3)
	parser.add_argument("--num_gen_synsp", type=int, default=3)
	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	#we use the same learning rate that was found optimal in the highresource setting for each task.
	task_to_lr = {'rte': 2e-5,
				'mrpc': 3e-5,
				'stsb': 4e-5,
				'sst2': 2e-5,
				'cola': 2e-5,
				'qqp': 2e-5,
				'mnli': 2e-5,
				'qnli': 2e-5}

	args.learning_rate = task_to_lr.get(args.task_name)
	torch.cuda.set_device(args.gpu)
	set_seed(args.seed)
	args.output_dir = os.path.join(args.output_dir, args.task_name)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	#Eyettention model setup
	cf = {"model_pretrained": "bert-base-cased",
			"used_sn_len":24,
			"max_sn_len": 256, #include start token and end token,
			#Here we count the maximum number of tokens a sentence includes.
			"max_sn_token": 512, #include start token and end token,
			"max_pred_len": 256,
			"remove_punctuation_split": False
			}

	if args.task_name in ['sst2', 'mrpc']:
		cf["remove_punctuation_split"]=True

	if args.task_name in ['mnli', 'qnli']:
		cf["max_sn_len"] = 400
		cf["max_pred_len"] = 350

	cf["tau"] = args.tau
	args.max_sn_len = cf['max_sn_len']

	#Encode the label into interger categories, setting the exclusive category 'cf["max_sn_len"]-1' as the end sign
	le = LabelEncoder()
	le.fit(np.append(np.arange(-cf["used_sn_len"]+3, cf["used_sn_len"]-1), cf["used_sn_len"]-1))
	#le.classes_

	# download the dataset.
	raw_datasets = load_dataset("glue", args.task_name)
	# Labels
	is_regression = args.task_name == "stsb"
	if not is_regression:
		label_list = raw_datasets["train"].features["label"].names
		num_labels = len(label_list)
	else:
		num_labels = 1
	args.num_labels = num_labels

	# Preprocessing the datasets
	sentence1_key, sentence2_key = task_to_keys[args.task_name]

	#load tokenizer
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

	def preprocess_function(examples, cf=cf):
		examples_ori = copy.deepcopy(examples)
		res = dict()

		#We need to prepare two sets of inputs, one is the NLP language model (BERT) and the other is the Eyettention model.
		#The two sets of inputs are different for the paired-sentence data sets.
		#BERT concatenates the two sentences and separates them with SEP tokens.
		#For Eyettention, we predict the scanpath for each single sentence separately, and then concatenate

		# for Eyettention model input
		texts = (
			(examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
		)

		#tokenization and padding
		#we manually add CLS and SEP tokens since we want the word_ids to start from 0 for CLS and end with the last number for SEP
		for s in range(len(texts)):
			for idx in range(len(texts[s])):
				if cf["remove_punctuation_split"] == True:
					#Scanpath prediction is word-based, punctuation marks and their adjacent words are treated as one word
					#so we remove the space between punctuation marks and words in the dataset.
					txt = remove_punctuation_split(texts[s][idx])
				else:
					txt = texts[s][idx]
				texts[s][idx] = ('[CLS]' + ' ' + txt + ' ' + '[SEP]').split()

		for s in range(len(texts)):#save each sentence separately
			#pre-tokenized input
			result = tokenizer(*(texts[s],), add_special_tokens = False, truncation=True, max_length = cf['max_sn_token'], padding = 'max_length', is_split_into_words=True)

			#use offset mapping to determine if two tokens are in the same word.
			#index start from 0, CLS -> 0 and SEP -> last index
			word_ids_list = []
			for i in range(len(result['input_ids'])):
				word_ids = result.word_ids(i)
				word_ids = [val if val is not None else np.nan for val in word_ids]
				word_ids_list.append(word_ids)
			result["word_ids"] = word_ids_list

			#prepare word length for the scanpath generation model
			word_len_list = []
			for idx in range(len(texts[s])):
				text_word_len = compute_word_length(texts[s][idx])
				word_len_list.append(text_word_len)

			word_len_list = pad_seq(word_len_list, args.max_sn_len, fill_value=np.nan, dtype=np.float32)
			result["word_len"] = word_len_list

			#update to final dictionary
			if s == 0:
				sentence_key = sentence1_key
			elif s == 1:
				sentence_key = sentence2_key
			for key in result.keys():
				res[sentence_key + '_' + key] = result[key]

		#for NLP model
		# Tokenize the texts
		texts = (
			(examples_ori[sentence1_key],) if sentence2_key is None else (examples_ori[sentence1_key], examples_ori[sentence2_key])
		)

		#tokenization and padding
		#Remove redundant punctuation separating spaces so that word IDs can be correctly mapped to gaze fixed words
		#we manually add CLS and SEP tokens since we want the word_ids to start from 0 for CLS and end with the last number for SEP
		for idx in range(len(texts[0])):
			for s in range(len(texts)):
				if cf["remove_punctuation_split"] == True:
					txt = remove_punctuation_split(texts[s][idx])
				else:
					txt = texts[s][idx]

				if s==0:
					texts[s][idx] = ('[CLS]' + ' ' + txt + ' ' + '[SEP]').split()
				elif s==1:
					texts[s][idx] = (txt + ' ' + '[SEP]').split()

		result = tokenizer(*texts, add_special_tokens = False, padding="max_length", max_length=args.max_length, truncation=True, is_split_into_words=True)
		#prepare word ids
		word_ids_list = []
		for i in range(len(result['input_ids'])):
			word_ids = result.word_ids(i)
			word_ids = [val if val is not None else np.nan for val in word_ids]

			SEP_word_id = word_ids[result['input_ids'][i].index(102)]
			#Make two sentences with consecutive word IDs
			if sentence2_key is not None:
				sn2_word_id = word_ids[word_ids.index(SEP_word_id)+1 :]
				sn2_word_id = [i+1+SEP_word_id for i in sn2_word_id]
				word_ids[word_ids.index(SEP_word_id)+1 :] = sn2_word_id

			word_ids_list.append(word_ids)

		result["word_ids"] = word_ids_list

		for key in result.keys():
			res['NLP_model_' + key] = result[key]

		if "label" in examples:
			# In all cases, rename the column to labels because the model will expect that.
			res["labels"] = examples["label"]

		res["sn_id"] = examples['idx']

		return res


	processed_datasets = raw_datasets.map(
		preprocess_function,
		batched=True,
		remove_columns=raw_datasets["train"].column_names,
		desc="Running tokenizer on dataset",
	)


	train_dataset = processed_datasets["train"]
	print(f'shuffling training set w. seed {args.data_seed}!')
	#We use the first K samples as the new training set, and the subsequent 1,000 samples as the development set.
	train_dataset_all = train_dataset.shuffle(seed=args.data_seed)
	train_dataset = train_dataset_all.select(range(args.max_train_samples))
	eval_dataset = train_dataset_all.select(range(args.max_train_samples, args.max_train_samples + 1000))
	test_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]


	# DataLoaders creation:
	num_gen_synsp = args.num_gen_synsp
	train_dataset = ConcatDataset([train_dataset]*num_gen_synsp)
	train_dataloaderr = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, drop_last=True)
	print(len(train_dataset))

	eval_dataset = ConcatDataset([eval_dataset]*num_gen_synsp)
	eval_dataloaderr = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size, drop_last=False)

	test_dataset = ConcatDataset([test_dataset]*num_gen_synsp)
	test_dataloaderr = DataLoader(test_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size, drop_last=False)

	#load sn_word_len mean and std from the pretrained model
	saved_res_path = 'feature_norm_celer.pickle'
	file_to_read = open(saved_res_path, "rb")
	loaded_dictionary = pickle.load(file_to_read)
	sn_word_len_mean = loaded_dictionary['sn_word_len_mean']
	sn_word_len_std = loaded_dictionary['sn_word_len_std']

	gaze_LM = PLM_AS(args, seed=args.seed)
	sp_gen_model = Eyettention(cf)
	sp_gen_model.load_state_dict(model_zoo.load_url('https://github.com/aeye-lab/EMNLP-SyntheticScanpaths-NLU-PretrainedLM/releases/download/v1.0/Eyettention_english.pth', map_location='cpu'))
	#freeze the parameters in scanpath generation model
	#for param in sp_gen_model.parameters():
		#param.requires_grad = False
	model = Joint_Gaze_LM(sp_gen_model, gaze_LM, args)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

	# Get the metric function
	if args.task_name is not None:
		metric = evaluate.load("glue", args.task_name)
	else:
		metric = evaluate.load("accuracy")

	starting_epoch = 0
	av_score = deque(maxlen=100)
	old_score = -10
	model.train()
	model.to(args.gpu)
	loss_dict = {'train_loss':[], 'eval_res':[], 'test_res':[]}
	print('Start training')
	for epoch in range(starting_epoch, args.num_train_epochs):
		print('episode:', epoch)
		model.train()
		print(count_parameters(model))
		counter = 0
		for step, batch in enumerate(train_dataloaderr):
			counter += 1

			batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}

			# #normalize word length feature
			batch[f'{sentence1_key}_word_len'] = (batch[f'{sentence1_key}_word_len'] - sn_word_len_mean)/sn_word_len_std
			batch[f'{sentence1_key}_word_len'] = torch.nan_to_num(batch[f'{sentence1_key}_word_len'])
			if sentence2_key is not None:
				batch[f'{sentence2_key}_word_len'] = (batch[f'{sentence2_key}_word_len'] - sn_word_len_mean)/sn_word_len_std
				batch[f'{sentence2_key}_word_len'] = torch.nan_to_num(batch[f'{sentence2_key}_word_len'])

			logits = model(batch, le, task_to_keys[args.task_name])
			labels = batch['labels']
			loss = None
			if num_labels == 1:
				#  We are doing regression
				loss_fct = MSELoss()
				loss = loss_fct(logits.view(-1), labels.view(-1))
			else:
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			av_score.append(loss.to('cpu').detach().numpy())
			print('counter:',counter)
			print('\rSample {}\tAverage Error: {:.10f} '.format(counter, np.mean(av_score)), end=" ")


		loss_dict['train_loss'].append(np.mean(av_score))

		#val_loss = []
		model.eval()
		ref = []
		pred_logits = []
		sn_id_list = []
		for step, batch in enumerate(eval_dataloaderr):
			with torch.no_grad():
				sn_id_list.extend(batch['sn_id'].numpy().tolist())

				batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}

				# #normalize word length feature
				batch[f'{sentence1_key}_word_len'] = (batch[f'{sentence1_key}_word_len'] - sn_word_len_mean)/sn_word_len_std
				batch[f'{sentence1_key}_word_len'] = torch.nan_to_num(batch[f'{sentence1_key}_word_len'])
				if sentence2_key is not None:
					batch[f'{sentence2_key}_word_len'] = (batch[f'{sentence2_key}_word_len'] - sn_word_len_mean)/sn_word_len_std
					batch[f'{sentence2_key}_word_len'] = torch.nan_to_num(batch[f'{sentence2_key}_word_len'])

				logits = model(batch, le, task_to_keys[args.task_name])
				logits = logits.view(-1, num_labels)
				references = batch["labels"]

				ref.extend(references.detach().cpu().numpy().tolist())
				pred_logits.append(logits.detach().cpu().numpy())

		pred_logits = np.concatenate(pred_logits)

		#merge logit scores for multiple scanpath on the same sentence
		avg_logits = []
		references = []
		for sn_id in np.unique(sn_id_list):
			score_indx = np.where(np.array(sn_id_list) == sn_id)[0]
			avg_score = pred_logits[score_indx, :].mean(0)
			avg_logits.append(avg_score[None, :])
			references.append(ref[score_indx[0]])

		avg_logits = np.concatenate(avg_logits)
		predictions = avg_logits.argmax(axis=-1) if not is_regression else avg_logits.squeeze()
		metric.add_batch(
					predictions=predictions,
					references=references,
				)

		eval_metric = metric.compute()
		loss_dict['eval_res'].append(eval_metric)

		new_score = eval_metric.get(task_to_metric.get(args.task_name))
		print('\nEVAL score is {} \n'.format(loss_dict['eval_res']))

		if new_score > old_score:
			# save model if val loss is smallest
			torch.save(model.state_dict(), '{}/State_glues_ours_numsynsp{}_dataseed{}_trainsample{}.pth'.format(args.output_dir, args.task_name, args.num_gen_synsp, args.data_seed, args.max_train_samples))
			old_score= new_score
			print('\nsaved model state dict\n')
			save_ep_couter = epoch
		else:
			#early stopping
			if epoch - save_ep_couter >= args.earlystop_patience:
				break

	loss_dict['save_ep'] = save_ep_couter

	model.eval()
	model.load_state_dict(torch.load(f'{args.output_dir}/State_glue_ours_numsynsp{args.num_gen_synsp}_dataseed{args.data_seed}_trainsample{args.max_train_samples}.pth', map_location='cpu'))
	model.to(args.gpu)
	ref = []
	pred_logits = []
	sn_id_list = []
	for step, batch in enumerate(test_dataloaderr):
		with torch.no_grad():
			sn_id_list.extend(batch['sn_id'].numpy().tolist())

			batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}

			# #normalize word length feature
			batch[f'{sentence1_key}_word_len'] = (batch[f'{sentence1_key}_word_len'] - sn_word_len_mean)/sn_word_len_std
			batch[f'{sentence1_key}_word_len'] = torch.nan_to_num(batch[f'{sentence1_key}_word_len'])
			if sentence2_key is not None:
				batch[f'{sentence2_key}_word_len'] = (batch[f'{sentence2_key}_word_len'] - sn_word_len_mean)/sn_word_len_std
				batch[f'{sentence2_key}_word_len'] = torch.nan_to_num(batch[f'{sentence2_key}_word_len'])

			logits = model(batch, le, task_to_keys[args.task_name])
			logits = logits.view(-1, num_labels)
			references = batch["labels"]

			ref.extend(references.detach().cpu().numpy().tolist())
			pred_logits.append(logits.detach().cpu().numpy())

	pred_logits = np.concatenate(pred_logits)

	#merge logit scores for multiple scanpath on the same sentence
	avg_logits = []
	references = []
	for sn_id in np.unique(sn_id_list):
		score_indx = np.where(np.array(sn_id_list) == sn_id)[0]
		avg_score = pred_logits[score_indx, :].mean(0)
		avg_logits.append(avg_score[None, :])
		references.append(ref[score_indx[0]])

	avg_logits = np.concatenate(avg_logits)
	predictions = avg_logits.argmax(axis=-1) if not is_regression else avg_logits.squeeze()
	metric.add_batch(
				predictions=predictions,
				references=references,
			)
	test_metric = metric.compute()
	loss_dict['test_res'].append(test_metric)
	print('\nTest score is {} \n'.format(loss_dict['test_res']))
	#save results
	with open(f'{args.output_dir}/res_GLUE_ours_numsyn{args.num_gen_synsp}_dataseed{args.data_seed}_trainsample{args.max_train_samples}.pickle', 'wb') as handle:
		pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
	main()
