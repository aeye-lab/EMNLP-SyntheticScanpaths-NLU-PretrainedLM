#coding=utf-8
import numpy as np
import pandas as pd
import os
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertTokenizer, BertForTokenClassification
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from collections import Counter
import torch.nn as nn
from ast import literal_eval



def load_corpus(corpus, task=None):
	if corpus == 'ET_Sentiment1':
		#TODO
		return None
	elif corpus == 'ET_Sentiment2':
		fix_seq_df = pd.read_csv('data/Eye-Tracking-Sentiment-AnalysisII/Fixation_sequence.csv')
		text_info_df = pd.read_csv('data/Eye-Tracking-Sentiment-AnalysisII/text_and_annorations.csv')
		return text_info_df, fix_seq_df

def _process_ET_Sentiment2_corpus(sn_list, reader_list, text_info_df, fix_seq_df, tokenizer, args):
	sn_id_list = []
	input_ids_list = []
	attention_mask_list = []
	token_type_ids_list = []
	word_ids_list = []
	labels_list = []
	gaze_wordpos_list = []
	gaze_pos_list = []
	gaze_dur_list = []
	word_len_list = []

	for sn_id in tqdm(sn_list):
		sn_df = text_info_df[text_info_df.Text_ID==sn_id]
		sn_str = sn_df.Text.values[0]
		#aspect information is shown at the beginning of the sentence
		aspect = sn_df.Aspect.values[0]
		sn_str = 'Aspect--' + aspect + ' ' + sn_str

		#tokenization and padding
		txt = ('[CLS]' + ' ' + sn_str + ' ' + '[SEP]').split()

		#pre-tokenized input
		tokens = tokenizer.encode_plus(txt,
										add_special_tokens = False,
										padding="max_length",
										max_length=args.max_length,
										truncation=True,
										is_split_into_words=True)

		#use offset mapping to determine if two tokens are in the same word.
		#index start from 0, CLS -> 0 and SEP -> last index
		word_ids_sn = tokens.word_ids()
		word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

		#for Eyettention model
		if hasattr(args, 'max_sn_len'):
			word_len = [compute_word_length(txt)]
			word_len = pad_seq(word_len, args.max_sn_len, fill_value=np.nan, dtype=np.float32).squeeze()


		label = sn_df.Default_Polarity.values[0]
		if label == -1:
			label = 0

		for sub_id in reader_list:
			sub_df = fix_seq_df[(fix_seq_df.Text_ID == sn_id) & (fix_seq_df.Participant_ID==sub_id)]
			if sub_df.size == 0:
				continue

			gaze_pos = sub_df.Word_ID.values
			gaze_token_pos = [np.where(np.array(word_ids_sn)==pos)[0].tolist() for pos in gaze_pos]
			#flatten the list
			gaze_token_pos = [item for sublist in gaze_token_pos for item in sublist]
			#padding
			gaze_token_pos.extend([args.max_length-1] * (args.max_length-len(gaze_token_pos)))

			sn_id_list.append(sn_id)
			input_ids_list.append(tokens['input_ids'])
			attention_mask_list.append(tokens['attention_mask'])
			token_type_ids_list.append(tokens['token_type_ids'])
			labels_list.append(label)
			gaze_pos_list.append(gaze_token_pos)
			if hasattr(args, 'max_sn_len'):
				word_len_list.append(word_len)
				word_ids_list.append(word_ids_sn)


	input_ids_list = np.asarray(input_ids_list, dtype=np.int64)
	attention_mask_list = np.asarray(attention_mask_list, dtype=np.int64)
	token_type_ids_list = np.asarray(token_type_ids_list, dtype=np.int64)
	labels_list = np.asarray(labels_list)
	sn_id_list = np.asarray(sn_id_list)
	gaze_pos_list = np.asarray(gaze_pos_list, dtype=np.float64)

	data = {"input_ids": input_ids_list,
			"attention_mask": attention_mask_list,
			"token_type_ids": token_type_ids_list,
			"labels": labels_list,
			"sn_id": sn_id_list,
			"gaze_pos": gaze_pos_list
			}

	if hasattr(args, 'max_sn_len'):
		word_len_list = np.asarray(word_len_list)
		data["word_len"] = word_len_list

		word_ids_list = np.asarray(word_ids_list)
		data["word_ids"] = word_ids_list

	return data

class ET_Sentiment2_Dataset(Dataset):
	"""Return Sentences and Scanpaths from Eye Tracking Sentiment analysis 2 dataset."""

	def __init__(self, text_info_df, fix_seq_df, sn_list, reader_list, tokenizer, args):
		self.data = _process_ET_Sentiment2_corpus(sn_list, reader_list, text_info_df, fix_seq_df, tokenizer, args)
		self.args = args
	def __len__(self):
		return len(self.data["input_ids"])

	def __getitem__(self,idx):
		sample = {}
		sample["input_ids"] = self.data["input_ids"][idx,:]
		sample["attention_mask"] = self.data["attention_mask"][idx,:]
		sample['token_type_ids'] = self.data['token_type_ids'][idx,:]
		sample["labels"] = self.data["labels"][idx]
		sample["sn_id"] = self.data["sn_id"][idx]
		sample["gaze_pos"] = self.data["gaze_pos"][idx,:]
		#for Eyettention
		if hasattr(self.args, 'max_sn_len'):
			sample["word_ids"] = self.data["word_ids"][idx,:]
			sample["word_len"] = self.data["word_len"][idx,:]
		return sample



def compute_word_length(txt):
	txt_word_len = [len(t) for t in txt[1:-1]]
	#pad nan for CLS and SEP tokens
	txt_word_len = [np.nan] + txt_word_len + [np.nan]
	#length of a punctuation is 0, plus an epsilon to avoid division output inf
	arr = np.array(txt_word_len).astype('float64')
	arr[arr==0] = 1/(0+0.5)
	arr[arr!=0] = 1/(arr[arr!=0])
	return arr

def pad_seq(seqs, max_len, dtype=np.long, fill_value=np.nan):
	padded = np.full((len(seqs), max_len), fill_value=fill_value, dtype=dtype)
	for i, seq in enumerate(seqs):
		padded[i, :len(seq)] = seq
	return padded

class ET_Sentiment2_Textonly_Dataset(Dataset):
	"""Return Sentences only from the Eye Tracking Sentiment analysis 2 dataset. Ignore scanpaths."""

	def __init__(self, text_info_df, sn_list, tokenizer, args):
		self.args = args
		text_info_df = text_info_df[text_info_df.Text_ID.isin(sn_list)]
		texts = text_info_df.Text.values.tolist()
		#aspect information is shown at the beginning of the sentence
		aspect = text_info_df.Aspect.values
		texts = [('[CLS]' + ' ' + 'Aspect--' + aspect[index] + ' ' + txt + ' ' + '[SEP]').split() for index, txt in enumerate(texts)]

		if hasattr(args, 'max_sn_len'):
			text_word_len = [compute_word_length(txt) for txt in texts]
			text_word_len = pad_seq(text_word_len, args.max_sn_len, fill_value=np.nan, dtype=np.float32)

		texts = ((texts,))
		result = tokenizer(*texts, add_special_tokens = False, padding="max_length", max_length=args.max_length, truncation=False, is_split_into_words=True)

		word_ids_list = []
		for i in range(len(result['input_ids'])):
			word_ids = result.word_ids(i)
			word_ids = [val if val is not None else np.nan for val in word_ids]
			word_ids_list.append(word_ids)
		result["word_ids"] = word_ids_list

		result["sn_id"] = text_info_df.Text_ID.values
		result['labels'] = text_info_df.Default_Polarity.values
		result['labels'][result['labels'] == -1] = 0
		if hasattr(args, 'max_sn_len'):
			result['word_len'] = text_word_len

		result = {k: np.array(v) for k, v in result.items()}
		self.data = result

	def __len__(self):
		return len(self.data["input_ids"])

	def __getitem__(self,idx):
		sample = {}
		sample['input_ids'] = self.data['input_ids'][idx,:]
		sample['attention_mask'] = self.data['attention_mask'][idx,:]
		sample['token_type_ids'] = self.data['token_type_ids'][idx,:]
		sample['word_ids'] = self.data['word_ids'][idx,:]
		sample['sn_id'] = self.data['sn_id'][idx]
		sample['labels'] = self.data['labels'][idx]
		if hasattr(self.args, 'max_sn_len'):
			sample['word_len'] = self.data['word_len'][idx,:]
		return sample



def load_position_label(sp_pos, cf, labelencoder, device):
	#prepare label and mask
	pad_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"])
	end_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"]-1)
	mask = pad_mask + end_mask
	sac_amp = sp_pos[:, 1:] - sp_pos[:, :-1]
	label = sp_pos[:, 1:]*mask + sac_amp*~mask
	label = torch.where(label>cf["max_sn_len"]-1, cf["max_sn_len"]-1, label).to('cpu').detach().numpy()
	label = labelencoder.transform(label.reshape(-1)).reshape(label.shape[0], label.shape[1])
	if device == 'cpu':
		pad_mask = pad_mask.to('cpu').detach().numpy()
	else:
		label = torch.from_numpy(label).to(device)
	return pad_mask, label

def gradient_clipping(dnn_model, clip = 10):
	torch.nn.utils.clip_grad_norm_(dnn_model.parameters(),clip)


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_mean_std(dataloader, feat_key, padding_value=0, scale=1):
	#calculate mean
	total_sum = 0
	total_num = 0
	for batchh in dataloader:
		batchh.keys()
		feat = batchh[feat_key]/scale
		feat = torch.nan_to_num(feat)
		total_num += len(feat.view(-1).nonzero())
		total_sum += feat.sum()
	feat_mean = total_sum / total_num
	#calculate std
	sum_of_squared_error = 0
	for batchh in dataloader:
		batchh.keys()
		feat = batchh[feat_key]/scale
		feat = torch.nan_to_num(feat)
		mask = ~torch.eq(feat, padding_value)
		sum_of_squared_error += (((feat - feat_mean).pow(2))*mask).sum()
	feat_std = torch.sqrt(sum_of_squared_error / total_num)
	return feat_mean, feat_std

def celer_load_L1_data_list():
    sub_metadata_path = 'INSERT YOUR DATA PATH HERE' + 'metadata.tsv'
    sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
    native_sub_list = sub_infor[sub_infor.L1 == 'English'].List.values.tolist()

    load_path= 'INSERT YOUR DATA PATH HERE' #csv file
    data_df = pd.read_csv(load_path, sep='\t')
    sn_list = np.unique(data_df[data_df['Sub_ID'].isin(native_sub_list)].SN_ID.values).tolist()
    return native_sub_list, sn_list



class celerdataset(Dataset):
	"""Return celer dataset."""

	def __init__(
		self,
		cf, reader_list, sn_list, tokenizer
	):
		self.sn_list = sn_list
		self.reader_list = reader_list
		self.data = _process_celer(self.sn_list, self.reader_list, tokenizer, cf)
		print(len(self.data["SN"]))
	def __len__(self):
		return len(self.data["SN"])


	def __getitem__(self,idx):
		sample = {}
		sample["sn"] = self.data["SN"][idx,:]
		sample["sn_mask"] = self.data["SN_mask"][idx,:]
		sample["sp_token"] = self.data["SP_token"][idx,:]
		sample["sp_token_mask"] = self.data["SP_token_mask"][idx,:]
		sample["sp_pos"] = self.data["SP_word_index"][idx,:]
		sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
		sample['word_ids_sn'] =  self.data['WORD_ids_sn'][idx,:]
		sample['word_ids_sp'] =  self.data['WORD_ids_sp'][idx,:]
		sample["SN_WORD_len"] = self.data['SN_WORD_len'][idx,:]
		return sample

def _process_celer(sn_list, reader_list, tokenizer, cf):
	"""
	SN embedding   <CLS>, bla, bla, <SEP>
	SP_token       <CLS>, bla, bla, <SEP>
	SP_ordinal_pos 0, bla, bla, max_sp_len
	SP_fix_dur     0, bla, bla, 0
	"""
	load_path= 'INSERT YOUR DATA PATH HERE' #csv file
	data_df = pd.read_csv(load_path, sep='\t')

	SN, SN_mask, SN_len, SP_token, SP_token_mask, SP_word_index, SP_landing_pos, SP_fix_dur, SP_len = [], [], [], [], [], [], [], [], []
	WORD_ids_sn, WORD_ids_sp = [], []
	sub_id_list =  []
	SN_WORD_len = []

	max_sn_len = 24
	max_sn_token = 35
	max_sp_len = 52
	max_sp_token = 395
	for sn_id in tqdm(sn_list):
		sn_df = data_df[data_df.SN_ID == sn_id]
		sn_word_len = literal_eval(sn_df.iloc[0].word_len)
		sn_str = sn_df.iloc[0].SN_str
		sn_len = sn_df.iloc[0].SN_len
		sn_token_len = sn_df.iloc[0].SN_token_len
		if sn_token_len + 2 > max_sn_token:
			max_sn_token = sn_token_len + 2
		if sn_len + 2 > max_sn_len:
			max_sn_len = sn_len +2

		#tokenization and padding
		tokenizer.padding_side = 'right'
		sn_str = '[CLS]' + ' ' + sn_str + ' ' + '[SEP]'
		#pre-tokenized input
		tokens = tokenizer.encode_plus(sn_str.split(), add_special_tokens = False, truncation=False, max_length = cf['max_sn_token'], padding = 'max_length', return_attention_mask=True, is_split_into_words=True)
		encoded_sn = tokens['input_ids']
		mask_sn = tokens['attention_mask']
		#use offset mapping to determine if two tokens are in the same word.
		#index start from 0, CLS -> 0 and SEP -> last index
		word_ids_sn = tokens.word_ids()
		word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

		#select L1 data
		sn_df = sn_df[sn_df['Sub_ID'].isin(reader_list)]
		#process scanpath one by one
		for index, row in sn_df.iterrows():
			SP_word_index.append(literal_eval(row.SP_word_index))
			SP_fix_dur.append(literal_eval(row.SP_fix_dur))
			SP_landing_pos.append(literal_eval(row.SP_landpos.replace('nan', 'None')))

			sp_token = [sn_str.split()[int(i)] for i in literal_eval(row.SP_word_index)]
			sp_token_str = '[CLS]' + ' ' + ' '.join(sp_token) + ' ' + '[SEP]'
			sp_token_len = len(tokenizer.tokenize(sp_token_str))
			sp_len = row.SP_len + 2
			if sp_token_len > max_sp_token:
				max_sp_token = sp_token_len
			if sp_len > max_sp_len:
				max_sp_len = sp_len
			#tokenization and padding
			#tokenizer.padding_side = 'right'
			sp_tokens = tokenizer.encode_plus(sp_token_str.split(), add_special_tokens = False, truncation=False, max_length = cf['max_sp_token'], padding = 'max_length', return_attention_mask=True, is_split_into_words=True)
			encoded_sp = sp_tokens['input_ids']
			mask_sp = sp_tokens['attention_mask']
			#index start from 0, CLS -> 0 and SEP -> last index
			word_ids_sp = sp_tokens.word_ids()
			word_ids_sp = [val if val is not None else np.nan for val in word_ids_sp]
			SP_token.append(encoded_sp)
			WORD_ids_sn.append(word_ids_sn)
			WORD_ids_sp.append(word_ids_sp)
			SP_token_mask.append(mask_sp)

			#prepare encoder input
			SN.append(encoded_sn)
			SN_mask.append(mask_sn)
			#SN_len.append(sn_len)
			sub_id_list.append(row.Sub_ID)
			SN_WORD_len.append(sn_word_len)

	#SP_fix_dur
	SP_word_index = pad_seq_for_celer(SP_word_index, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
	SP_fix_dur = pad_seq_for_celer(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
	#min: 50ms, max: 5000ms
	SP_landing_pos = pad_seq_for_celer(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
	SN_WORD_len = pad_seq_with_nan_for_celer(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)
	print('max_sn_len:', max_sn_len)
	print('max_sn_token:', max_sn_token)
	print('max_sp_len:', max_sp_len)
	print('max_sp_token:', max_sp_token)

	#max: 6.33
	SN = np.asarray(SN, dtype=np.int64)
	SN_mask = np.asarray(SN_mask, dtype=np.float32)
	SP_token = np.asarray(SP_token, dtype=np.int64)
	SP_token_mask = np.asarray(SP_token_mask, dtype=np.float32)
	sub_id_list = np.asarray(sub_id_list, dtype=np.int64)
	WORD_ids_sn = np.asarray(WORD_ids_sn)
	WORD_ids_sp = np.asarray(WORD_ids_sp)
	data = {"SN": SN,
			"SN_mask": SN_mask,
			#"SN_len": np.array(SN_len),
			"SP_token": SP_token,
			"SP_token_mask": SP_token_mask,
			"SP_word_index": np.array(SP_word_index),
			"SP_landing_pos": np.array(SP_landing_pos),
			"SP_fix_dur": np.array(SP_fix_dur),
			#"SP_len": np.array(SP_len),
			"sub_id": sub_id_list,
			"WORD_ids_sn": WORD_ids_sn,
			"WORD_ids_sp": WORD_ids_sp,
			"SN_WORD_len": SN_WORD_len}
	return data

def pad_seq_for_celer(seqs, max_len, pad_value, dtype=np.compat.long):
	padded = np.full((len(seqs), max_len), fill_value=pad_value, dtype=dtype)
	for i, seq in enumerate(seqs):
		padded[i, 0] = 0
		padded[i, 1:(len(seq)+1)] = seq
		if pad_value !=0:
			padded[i, len(seq)+1] = pad_value -1
	return padded


def load_position_label(sp_pos, cf, labelencoder, device):
	#prepare label and mask
	pad_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"])
	end_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"]-1)
	mask = pad_mask + end_mask
	sac_amp = sp_pos[:, 1:] - sp_pos[:, :-1]
	label = sp_pos[:, 1:]*mask + sac_amp*~mask
	label = torch.where(label>cf["max_sn_len"]-1, cf["max_sn_len"]-1, label).to('cpu').detach().numpy()
	label = labelencoder.transform(label.reshape(-1)).reshape(label.shape[0], label.shape[1])
	if device == 'cpu':
		pad_mask = pad_mask.to('cpu').detach().numpy()
	else:
		label = torch.from_numpy(label).to(device)
	return pad_mask, label


def pad_seq_with_nan_for_celer(seqs, max_len, dtype=np.compat.long):
	padded = np.full((len(seqs), max_len), fill_value=np.nan, dtype=dtype)
	for i, seq in enumerate(seqs):
		padded[i, 1:(len(seq)+1)] = seq
	return padded

