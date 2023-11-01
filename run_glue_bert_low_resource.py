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

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
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

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
	parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
	parser.add_argument(
		"--gpu",
		type=int,
		default=3,
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
		"--per_device_train_batch_size",
		type=int,
		default=32,
		help="Batch size (per device) for the training dataloader.",
	)
	parser.add_argument(
		"--per_device_eval_batch_size",
		type=int,
		default=32,
		help="Batch size (per device) for the evaluation dataloader.",
	)

	parser.add_argument(
		"--max_train_samples",
		type=int,
		default=1000,
		help="truncate the number of training examples to this value if set.",
	)

	#parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
	parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
	parser.add_argument("--output_dir", type=str, default='glue_low_resource_results/', help="Where to store the final model.")
	parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
	parser.add_argument("--data_seed", type=int, default=111, help="A seed for reproducible training.")

	parser.add_argument("--earlystop_patience", type=int, default=3)
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
	args.output_dir = os.path.join(args.output_dir, args.task_name)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# download the dataset.
	raw_datasets = load_dataset("glue", args.task_name)
	# Labels
	is_regression = args.task_name == "stsb"
	if not is_regression:
		label_list = raw_datasets["train"].features["label"].names
		num_labels = len(label_list)
	else:
		num_labels = 1

	# Set seed before initializing model.
	set_seed(args.seed)
	# Load pretrained model and tokenizer
	# download model & vocab.
	config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
	model = AutoModelForSequenceClassification.from_pretrained(
		args.model_name_or_path,
		from_tf=bool(".ckpt" in args.model_name_or_path),
		config=config,
		ignore_mismatched_sizes=False,
	)

	# Preprocessing the datasets
	sentence1_key, sentence2_key = task_to_keys[args.task_name]

	# Some models have set the order of the labels to use, so let's make sure we do use it.
	label_to_id = None
	if (
		model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
		and args.task_name is not None
		and not is_regression
	):
		# Some have all caps in their config, some don't.
		label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
		if sorted(label_name_to_id.keys()) == sorted(label_list):
			logger.info(
				f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
				"Using it!"
			)
			label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
		else:
			logger.info(
				"Your model seems to have been trained with labels, but they don't match the dataset: ",
				f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
				"\nIgnoring the model labels as a result.",
			)
	elif args.task_name is None and not is_regression:
		label_to_id = {v: i for i, v in enumerate(label_list)}

	if label_to_id is not None:
		model.config.label2id = label_to_id
		model.config.id2label = {id: label for label, id in config.label2id.items()}
	elif args.task_name is not None and not is_regression:
		model.config.label2id = {l: i for i, l in enumerate(label_list)}
		model.config.id2label = {id: label for label, id in config.label2id.items()}



	def preprocess_function(examples):

		# Tokenize the texts
		texts = (
			(examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
		)
		result = tokenizer(*texts, padding="max_length", max_length=args.max_length, truncation=True)

		if "label" in examples:
			if label_to_id is not None:
				# Map labels to IDs (not necessary for GLUE tasks)
				result["labels"] = [label_to_id[l] for l in examples["label"]]
			else:
				# In all cases, rename the column to labels because the model will expect that.
				result["labels"] = examples["label"]

		return result

	processed_datasets = raw_datasets.map(
		preprocess_function,
		batched=True,
		remove_columns=raw_datasets["train"].column_names,
		desc="Running tokenizer on dataset",
	)


	train_dataset = processed_datasets["train"]
	print(f'shuffling training set w. seed {args.data_seed}!')
	train_dataset_all = train_dataset.shuffle(seed=args.data_seed)
	#We use the first K samples as the new training set, and the subsequent 1,000 samples as the development set.
	train_dataset = train_dataset_all.select(range(args.max_train_samples))
	eval_dataset = train_dataset_all.select(range(args.max_train_samples, args.max_train_samples + 1000))
	test_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

	# DataLoaders creation:
	train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, drop_last=True)
	eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, drop_last=False)
	test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, drop_last=False)

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
		for step, batch in enumerate(train_dataloader):
			counter += 1
			batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			av_score.append(loss.to('cpu').detach().numpy())
			print('counter:',counter)
			print('\rSample {}\tAverage Error: {:.10f} '.format(counter, np.mean(av_score)), end=" ")
		loss_dict['train_loss'].append(np.mean(av_score))

		model.eval()
		for step, batch in enumerate(eval_dataloader):
			with torch.no_grad():
				batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}
				outputs = model(**batch)
				#loss = outputs.loss
				predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
				references = batch["labels"]
				metric.add_batch(
					predictions=predictions,
					references=references,
				)
		eval_metric = metric.compute()
		loss_dict['eval_res'].append(eval_metric)
		new_score = [v for k, v in eval_metric.items()][0]
		print('\nEVAL score is {} \n'.format(loss_dict['eval_res']))

		if new_score > old_score:
			# save model if val loss is smallest
			torch.save(model.state_dict(), '{}/State_GLUE_finetune_BERTbase_lr{}_dataseed{}_trainsample{}.pth'.format(args.output_dir, args.learning_rate, args.data_seed, args.max_train_samples))
			old_score= new_score
			print('\nsaved model state dict\n')
			save_ep_couter = epoch
		else:
			#early stopping
			if epoch - save_ep_couter >= args.earlystop_patience:
				break
	loss_dict['save_ep'] = save_ep_couter


	model.eval()
	model.load_state_dict(torch.load(f'{args.output_dir}/State_GLUE_finetune_BERTbase_lr{args.learning_rate}_dataseed{args.data_seed}_trainsample{args.max_train_samples}.pth', map_location='cpu'))
	model.to(args.gpu)
	for step, batch in enumerate(test_dataloader):
		with torch.no_grad():
			batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}
			outputs = model(**batch)
			#loss = outputs.loss
			predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
			references = batch["labels"]
			metric.add_batch(
				predictions=predictions,
				references=references,
			)
	test_metric = metric.compute()
	loss_dict['test_res'].append(test_metric)
	print('\nTest score is {} \n'.format(loss_dict['test_res']))
	#save results
	with open(f'{args.output_dir}/GLUE_BERTbase_res_lr{args.learning_rate}_dataseed{args.data_seed}_trainsample{args.max_train_samples}.pickle', 'wb') as handle:
		pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
	main()
