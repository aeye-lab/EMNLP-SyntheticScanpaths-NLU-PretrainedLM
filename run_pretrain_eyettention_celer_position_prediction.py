import numpy as np
import pandas as pd
import os
from utils import celer_load_L1_data_list, calculate_mean_std, celerdataset, count_parameters, load_position_label, gradient_clipping
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
from transformers import BertTokenizerFast
from model import Eyettention_pretrain
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn.functional import cross_entropy, softmax
from collections import deque
import pickle
import json
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='run uniform baseline')
	parser.add_argument(
		'--test_mode',
		help='test mode: text or subject',
		type=str,
		default='text'
	)
	parser.add_argument(
		'--gpu',
		help='gpu index',
		type=int,
		default=7
	)
	args = parser.parse_args()
	test_mode = args.test_mode
	gpu = args.gpu

	#use FastTokenizer lead to warning -> The current process just got forked
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	torch.set_default_tensor_type('torch.FloatTensor')
	availbl = torch.cuda.is_available()
	print(torch.cuda.is_available())
	if availbl:
		device = f'cuda:{gpu}'
	else:
		device = 'cpu'
	print(device)
	torch.cuda.set_device(gpu)

	cf = {"model_pretrained": "bert-base-cased",
			"weight_decay": 1e-2,
			"lr": 1e-3,
			"max_grad_norm": 10,
			"n_epochs": 1000,
			"test_mode": test_mode,
			"n_folds": 5,
			"dataset": 'celer',
			"batch_size": 256,
			"max_sn_len": 24, #include start token and end token,
			#Here we count the maximum number of tokens a sentence includes.
			"max_sn_token": 35, #include start token and end token,
			"max_sp_len": 52, #include start token and end token
			#Here we count the maximum number of tokens a scanpath includes.
			"max_sp_token": 395, #include start token and end token
			"norm_type": 'z-score',
			"earlystop_patience": 20,
			"data_fold": "./results"
			}

	#Encode the label into interger categories, setting the exclusive category 'cf["max_sn_len"]-1' as the end sign
	le = LabelEncoder()
	le.fit(np.append(np.arange(-cf["max_sn_len"]+3, cf["max_sn_len"]-1), cf["max_sn_len"]-1))
	#le.classes_

	reader_list, sn_list = celer_load_L1_data_list()
	split_list = sn_list

	n_folds = cf["n_folds"]
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	fold_indx = 0
	for train_idx, val_idx in kf.split(split_list):
		loss_dict = {'val_loss':[], 'train_loss':[], 'test_loss':[]}
		list_train = [split_list[i] for i in train_idx]
		list_val = [split_list[i] for i in val_idx]

		#initialize tokenizer
		tokenizer = BertTokenizerFast.from_pretrained(cf['model_pretrained'])
		#Preparing batch data
		dataset_train = celerdataset(cf, reader_list, list_train, tokenizer)
		train_dataloaderr = DataLoader(dataset_train, batch_size = cf["batch_size"], shuffle = True, num_workers=0, drop_last=True)
		dataset_val = celerdataset(cf, reader_list, list_val, tokenizer)
		val_dataloaderr = DataLoader(dataset_val, batch_size = cf["batch_size"], shuffle = False, num_workers=0, drop_last=True)
		#dataset_test = celerdataset(cf, reader_list, list_test, tokenizer)
		#test_dataloaderr = DataLoader(dataset_test, batch_size = cf["batch_size"], shuffle = False, num_workers=0, drop_last=False)

		#z-score normalization for gaze features
		sn_word_len_mean, sn_word_len_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="SN_WORD_len")

		# load model here
		dnn = Eyettention_pretrain(cf)

		#training
		episode = 0
		optimizer = Adam(dnn.parameters(), lr=cf["lr"])
		dnn.train()
		dnn.to(device)
		av_score = deque(maxlen=100)
		old_score = 1e10
		save_ep_couter = 0
		print('Start training')
		for episode_i in range(episode, cf["n_epochs"]+1):
			dnn.train()
			print('episode:', episode_i)
			counter = 0
			for batchh in train_dataloaderr:
				counter += 1
				batchh.keys()
				sn_emd = batchh["sn"].to(device)
				sn_mask = batchh["sn_mask"].to(device)
				sp_emd = batchh["sp_token"].to(device)
				sp_mask = batchh["sp_token_mask"].to(device)
				sp_pos = batchh["sp_pos"].to(device)
				#sp_landing_pos = batchh["sp_landing_pos"].to(device)
				#sp_fix_dur = (batchh["sp_fix_dur"]/1000).to(device)
				word_ids_sn = batchh["word_ids_sn"].to(device)
				word_ids_sp = batchh["word_ids_sp"].to(device)
				sn_word_len = batchh["SN_WORD_len"].to(device)

				#normalize gaze features
				sn_word_len = (sn_word_len - sn_word_len_mean)/sn_word_len_std
				sn_word_len = torch.nan_to_num(sn_word_len)#nan for padding

				# zero old gradients
				optimizer.zero_grad()
				# predict output with DNN
				dnn_out = dnn(sn_emd,
								sn_mask,
								sp_emd,
								sp_pos,
								word_ids_sn,
								word_ids_sp,
								sn_word_len = sn_word_len)

				print(count_parameters(dnn))
				dnn_out = dnn_out.permute(0,2,1)              #[batch, dec_o_dim, step]
				#prepare label and mask
				pad_mask, label = load_position_label(sp_pos, cf, le, device)
				loss = nn.CrossEntropyLoss(reduction="none")
				batch_error = torch.mean(torch.masked_select(loss(dnn_out, label), ~pad_mask))
				# backpropagate loss
				batch_error.backward()
				# clip gradients
				gradient_clipping(dnn, cf["max_grad_norm"])
				#learn
				optimizer.step()
				av_score.append(batch_error.to('cpu').detach().numpy())
				print('counter:',counter)
				print('\rSample {}\tAverage Error: {:.10f} '.format(counter, np.mean(av_score)), end=" ")
			loss_dict['train_loss'].append(np.mean(av_score))

			val_loss = []
			dnn.eval()
			for batchh in val_dataloaderr:
				with torch.no_grad():
					sn_emd_val = batchh["sn"].to(device)
					sn_mask_val = batchh["sn_mask"].to(device)
					sp_emd_val = batchh["sp_token"].to(device)
					sp_mask_val = batchh["sp_token_mask"].to(device)
					sp_pos_val = batchh["sp_pos"].to(device)
					word_ids_sn_val = batchh["word_ids_sn"].to(device)
					word_ids_sp_val = batchh["word_ids_sp"].to(device)
					sn_word_len_val = batchh["SN_WORD_len"].to(device)
					#normalize gaze features
					sn_word_len_val = (sn_word_len_val - sn_word_len_mean)/sn_word_len_std
					sn_word_len_val = torch.nan_to_num(sn_word_len_val)

					dnn_out_val = dnn(sn_emd_val,
										sn_mask_val,
										sp_emd_val,
										sp_pos_val,
										word_ids_sn_val,
										word_ids_sp_val,
										sn_word_len = sn_word_len_val)
					dnn_out_val = dnn_out_val.permute(0,2,1)              #[batch, dec_o_dim, step

					#prepare label and mask
					pad_mask_val, label_val = load_position_label(sp_pos_val, cf, le, device)
					batch_error_val = torch.mean(torch.masked_select(loss(dnn_out_val, label_val), ~pad_mask_val))
					val_loss.append(batch_error_val.detach().to('cpu').numpy())
			print('\nvalidation loss is {} \n'.format(np.mean(val_loss)))
			loss_dict['val_loss'].append(np.mean(val_loss))

			if np.mean(val_loss) < old_score:
				# save model if val loss is smallest
				torch.save(dnn.state_dict(), '{}/CELoss_CELER_{}_eyettention_location_prediction_newloss_fold{}.pth'.format(cf["data_fold"], test_mode, fold_indx))
				old_score= np.mean(val_loss)
				print('\nsaved model state dict\n')
				save_ep_couter = episode_i
			else:
				#early stopping
				if episode_i - save_ep_couter >= cf["earlystop_patience"]:
					break

		loss_dict['sn_word_len_mean'] = sn_word_len_mean
		loss_dict['sn_word_len_std'] = sn_word_len_std
		#save results
		with open('{}/res_CELER_{}_eyettention_location_prediction_Fold{}.pickle'.format(cf["data_fold"], test_mode, fold_indx), 'wb') as handle:
			pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		fold_indx += 1
		break
