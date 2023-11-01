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
from torch.nn import CrossEntropyLoss
from utils import load_corpus, ET_Sentiment2_Textonly_Dataset, count_parameters, calculate_mean_std
from model import Eyettention, PLM_AS, Joint_Gaze_LM
from scipy.special import softmax
from sklearn import metrics

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
		"--learning_rate",
		type=float,
		default=1e-5,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument(
		"--max_train_samples",
		type=int,
		default=1000,
		help="truncate the number of training examples to this value if set.",
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
	parser.add_argument("--output_dir", type=str, default='results/', help="Where to store the final model.")
	parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
	parser.add_argument("--earlystop_patience", type=int, default=3)
	parser.add_argument("--num_gen_synsp", type=int, default=7)
	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	torch.cuda.set_device(args.gpu)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	#Eyettention model setup
	cf = {"model_pretrained": "bert-base-cased",
			"used_sn_len":24,
			"max_sn_len": 256, #include start token and end token,
			#Here we count the maximum number of tokens a sentence includes.
			"max_sn_token": 512, #include start token and end token,
			"max_pred_len": 256
			}
	args.max_sn_len = cf['max_sn_len']

	#Encode the label into interger categories, setting the exclusive category 'cf["max_sn_len"]-1' as the end sign
	le = LabelEncoder()
	le.fit(np.append(np.arange(-cf["used_sn_len"]+3, cf["used_sn_len"]-1), cf["used_sn_len"]-1))
	#le.classes_

	#setup for binary task training
	label_list = ['negative', 'positive']
	num_labels = len(label_list)
	args.num_labels = num_labels
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
	# Get the metric function
	metric_f1 = evaluate.load("f1")
	metric_acc = evaluate.load("accuracy")

	#load corpus
	text_info_df, fix_seq_df = load_corpus('ET_Sentiment2')
	sn_list = np.unique(text_info_df.Text_ID).tolist()
	split_list = sn_list

	n_folds = 10
	loss_dict = {'train_loss':[], 'val_loss':[], 'test_res':[]}
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
	fold_indx = 0
	for train_idx, test_idx in kf.split(split_list):
		print('Folder:', fold_indx)
		set_seed(args.seed + fold_indx)

		list_train = [split_list[i] for i in train_idx]
		list_test = [split_list[i] for i in test_idx]
		# create train validation split for training the models:
		kf_val = KFold(n_splits=n_folds, shuffle=True, random_state=0)
		for train_index, val_index in kf_val.split(list_train):
			# we only evaluate a single fold
			break
		list_train_net = [list_train[i] for i in train_index]
		list_val_net = [list_train[i] for i in val_index]

		#if low-resource setting, sample a subset of training sentences
		random.shuffle(list_train_net)
		list_train_net = list_train_net[:args.max_train_samples]
		list_train_net.sort()

		num_gen_synsp = args.num_gen_synsp
		dataset_train = ET_Sentiment2_Textonly_Dataset(text_info_df, list_train_net, tokenizer, args)
		dataset_train = ConcatDataset([dataset_train]*num_gen_synsp)
		train_dataloaderr = DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, drop_last=True)
		print(len(dataset_train))

		dataset_val = ET_Sentiment2_Textonly_Dataset(text_info_df, list_val_net, tokenizer, args)
		dataset_val = ConcatDataset([dataset_val]*num_gen_synsp)
		val_dataloaderr = DataLoader(dataset_val, batch_size = args.batch_size, shuffle = False, drop_last=True)

		dataset_test = ET_Sentiment2_Textonly_Dataset(text_info_df, list_test, tokenizer, args)
		dataset_test = ConcatDataset([dataset_test]*num_gen_synsp)
		test_dataloaderr = DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False, drop_last=False)

		#load sn_word_len mean and std from the pretrained Eyettention model
		saved_res_path = 'feature_norm_celer.pickle'
		file_to_read = open(saved_res_path, "rb")
		loaded_dictionary = pickle.load(file_to_read)
		sn_word_len_mean = loaded_dictionary['sn_word_len_mean']
		sn_word_len_std = loaded_dictionary['sn_word_len_std']

		gaze_LM = PLM_AS(args, seed=args.seed + fold_indx)
		sp_gen_model = Eyettention(cf)
		sp_gen_model.load_state_dict(model_zoo.load_url('https://github.com/aeye-lab/EMNLP-SyntheticScanpaths-NLU-PretrainedLM/releases/download/v1.0/Eyettention_english.pth', map_location='cpu'))
		# #freeze the parameters in scanpath generation model
		# for param in sp_gen_model.parameters():
		# 	param.requires_grad = False
		model = Joint_Gaze_LM(sp_gen_model, gaze_LM)

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

		#completed_steps = 0
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

				# #normalize word length feature
				batch['word_len'] = (batch['word_len'] - sn_word_len_mean)/sn_word_len_std
				batch['word_len'] = torch.nan_to_num(batch['word_len'])

				logits = model(batch, le)
				labels = batch['labels']
				loss = None
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

				loss.backward()
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
				#completed_steps += 1
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
					# #normalize word length feature
					batch['word_len'] = (batch['word_len'] - sn_word_len_mean)/sn_word_len_std
					batch['word_len'] = torch.nan_to_num(batch['word_len'])

					logits = model(batch, le)
					labels = batch['labels']
					loss = None
					loss_fct = CrossEntropyLoss()
					loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
					val_loss.append(loss.detach().to('cpu').numpy())

			print('\nvalidation loss is {} \n'.format(np.mean(val_loss)))
			loss_dict['val_loss'].append(np.mean(val_loss))
			if np.mean(val_loss) < old_score:
				# save model if val loss is smallest
				torch.save(model.state_dict(), '{}/CELoss_ETSA_ours_numsynsp{}_{}_fold{}.pth'.format(args.output_dir, args.num_gen_synsp, args.max_train_samples, fold_indx))
				old_score= np.mean(val_loss)
				print('\nsaved model state dict\n')
				save_ep_couter = epoch
			else:
				#early stopping
				if epoch - save_ep_couter >= args.earlystop_patience:
					break

		#evaluation
		model.eval()
		model.load_state_dict(torch.load(os.path.join(args.output_dir,f'CELoss_ETSA_ours_numsynsp{args.num_gen_synsp}_{args.max_train_samples}_fold{fold_indx}.pth'), map_location='cpu'))
		model.to(args.gpu)
		ref = []
		pred_logits = []
		sn_id_list = []
		for step, batch in enumerate(test_dataloaderr):
			with torch.no_grad():
				sn_id_list.extend(batch['sn_id'].numpy().tolist())
				batch.pop('sn_id')
				batch = {k: v.to(device=args.gpu, non_blocking=True) for k, v in batch.items()}

				# #normalize word length feature
				batch['word_len'] = (batch['word_len'] - sn_word_len_mean)/sn_word_len_std
				batch['word_len'] = torch.nan_to_num(batch['word_len'])

				logits = model(batch, le)
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
	with open(f'{args.output_dir}/res_ETSA_ours_numsynsp{args.num_gen_synsp}_{args.max_train_samples}.pickle', 'wb') as handle:
		pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
	main()
