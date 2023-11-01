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
from scipy.special import softmax
from sklearn.model_selection import KFold
from sklearn import metrics
from torch.nn import CrossEntropyLoss
from utils import load_corpus, ET_Sentiment2_Dataset, count_parameters
from model import ETSA_PLM_AS

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

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
	parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
	parser.add_argument(
		"--gpu",
		type=int,
		default=0,
		help="The GPU index that we want to use.",
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
		"--num_sub",
		type=int,
		default=7,
		help="number of real scanpaths that used for the model training and testing, max number is 7.",
	)

	parser.add_argument(
		"--learning_rate",
		type=float,
		default=1e-5,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
	parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")

	parser.add_argument(
		"--lr_scheduler_type",
		type=SchedulerType,
		default="linear",
		help="The scheduler type to use.",
		choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
	)
	parser.add_argument(
		"--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
	)
	parser.add_argument("--output_dir", type=str, default='./results/', help="Where to store the final model.")
	parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
	parser.add_argument("--earlystop_patience", type=int, default=3)
	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	torch.cuda.set_device(args.gpu)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	label_list = ['negative', 'positive']
	num_labels = len(label_list)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
	# Get the metric function
	metric_f1 = evaluate.load("f1")
	metric_acc = evaluate.load("accuracy")

	#load corpus
	text_info_df, fix_seq_df = load_corpus('ET_Sentiment2')
	reader_list = np.unique(fix_seq_df.Participant_ID.values).tolist()
	sn_list = np.unique(text_info_df.Text_ID).tolist()
	split_list = sn_list

	n_folds = 10
	loss_dict = {'train_loss':[], 'val_loss':[], 'test_res':[]}
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
	fold_indx = 0
	for train_idx, test_idx in kf.split(split_list):
		set_seed(args.seed + fold_indx)
		#sample a proportion of subjects
		sel_reader = np.random.choice(reader_list, size=args.num_sub, replace=False)
		sel_reader.sort()

		list_train = [split_list[i] for i in train_idx]
		list_test = [split_list[i] for i in test_idx]
		# create train validation split for training the models:
		kf_val = KFold(n_splits=n_folds, shuffle=True, random_state=0)
		for train_index, val_index in kf_val.split(list_train):
			# we only evaluate a single fold
			break
		list_train_net = [list_train[i] for i in train_index]
		list_val_net = [list_train[i] for i in val_index]

		dataset_train = ET_Sentiment2_Dataset(text_info_df, fix_seq_df, list_train_net, sel_reader, tokenizer, args)
		train_dataloaderr = DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, drop_last=True)
		print(len(dataset_train))
		dataset_val = ET_Sentiment2_Dataset(text_info_df, fix_seq_df, list_val_net, sel_reader, tokenizer, args)
		val_dataloaderr = DataLoader(dataset_val, batch_size = args.batch_size, shuffle = False, drop_last=True)

		dataset_test = ET_Sentiment2_Dataset(text_info_df, fix_seq_df, list_test, sel_reader, tokenizer, args)
		test_dataloaderr = DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False, drop_last=False)

		# Load pretrained model and tokenizer
		# download model & vocab.
		config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task='sst2')
		config.output_hidden_states=True
		init_model = AutoModelForSequenceClassification.from_pretrained(
			args.model_name_or_path,
			from_tf=bool(".ckpt" in args.model_name_or_path),
			config=config,
			ignore_mismatched_sizes=False,
		)

		model = ETSA_PLM_AS(init_model)

		# Optimizer
		# Split weights in two groups, one with weight decay and the other not.
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
				"weight_decay": args.weight_decay,
			},
			{
				"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
				"weight_decay": 0.0,
			},
		]
		optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
		lr_scheduler = get_scheduler(
			name=args.lr_scheduler_type,
			optimizer=optimizer,
			num_warmup_steps=args.num_warmup_steps,
			num_training_steps=args.num_train_epochs * len(train_dataloaderr),
		)

		starting_epoch = 0
		av_score = deque(maxlen=100)
		old_score = 1e10
		model.train()
		model.to(args.gpu)
		print('Start training')
		for epoch in range(starting_epoch, args.num_train_epochs):
			print('episode:', epoch)
			model.train()
			print(count_parameters(model))
			counter = 0
			for step, batch in enumerate(train_dataloaderr):
				counter += 1
				batch.pop('sn_id')

				batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}
				logits = model(batch)

				labels = batch['labels']
				loss = None
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

				loss.backward()
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()

				av_score.append(loss.to('cpu').detach().numpy())
				print('counter:',counter)
				print('\rSample {}\tAverage Error: {:.10f} '.format(counter, np.mean(av_score)), end=" ")
			loss_dict['train_loss'].append(np.mean(av_score))

			val_loss = []
			model.eval()
			for step, batch in enumerate(val_dataloaderr):
				with torch.no_grad():
					batch.pop('sn_id')

					batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}
					logits = model(batch)
					labels = batch['labels']
					loss = None
					loss_fct = CrossEntropyLoss()
					loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
					val_loss.append(loss.detach().to('cpu').numpy())

			print('\nvalidation loss is {} \n'.format(np.mean(val_loss)))
			loss_dict['val_loss'].append(np.mean(val_loss))
			if np.mean(val_loss) < old_score:
				# save model if val loss is smallest
				torch.save(model.state_dict(), '{}/CELoss_ETSA_PLM_AS_numsub{}_fold{}.pth'.format(args.output_dir, args.num_sub, fold_indx))
				old_score= np.mean(val_loss)
				print('\nsaved model state dict\n')
				save_ep_couter = epoch
			else:
				#early stopping
				if epoch - save_ep_couter >= args.earlystop_patience:
					break

		#evaluation
		model.eval()
		model.load_state_dict(torch.load(os.path.join(args.output_dir,f'CELoss_ETSA_PLM_AS_numsub{args.num_sub}_fold{fold_indx}.pth'), map_location='cpu'))
		model.to(args.gpu)
		ref = []
		pred_logits = []
		sn_id_list = []
		for step, batch in enumerate(test_dataloaderr):
			with torch.no_grad():
				sn_id_list.extend(batch['sn_id'].numpy().tolist())
				batch.pop('sn_id')
				batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}
				logits = model(batch)
				logits = logits.view(-1, num_labels)
				references = batch["labels"]

				ref.extend(references.detach().cpu().numpy().tolist())
				pred_logits.append(logits.detach().cpu().numpy())

		pred_logits = np.concatenate(pred_logits)

		#merge logit scores for 7 scanpath on the same sentence
		avg_logits = []
		references = []
		for sn_id in np.unique(sn_id_list):
			score_indx = np.where(np.array(sn_id_list) == sn_id)[0]
			avg_score = pred_logits[score_indx, :].mean(0)
			avg_logits.append(avg_score[None, :])
			references.append(ref[score_indx[0]])

		avg_logits = np.concatenate(avg_logits, axis=0)
		predictions = avg_logits.argmax(axis=-1)
		prob = softmax(avg_logits, axis=-1)

		metric_f1.add_batch(
			predictions=predictions,
			references=references,
		)
		metric_acc.add_batch(
			predictions=predictions,
			references=references,
		)
		fpr, tpr, thresholds = metrics.roc_curve(references, prob[:, 1])
		auc = metrics.auc(fpr, tpr)

		eval_metric_f1 = metric_f1.compute()
		eval_metric_acc = metric_acc.compute()
		loss_dict['test_res'].append(eval_metric_f1)
		loss_dict['test_res'].append(eval_metric_acc)
		loss_dict['test_res'].append({'AUC': auc})
		print('\nTest score is {} \n'.format(eval_metric_f1))
		print('\nTest score is {} \n'.format(eval_metric_acc))
		print('\nTest score is AUC: {} \n'.format(auc))
		fold_indx += 1

	#save results
	with open(f'{args.output_dir}/res_PLM_AS_ETSA_numsub{args.num_sub}.pickle', 'wb') as handle:
		pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)






if __name__ == "__main__":
	main()
