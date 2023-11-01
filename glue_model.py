import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.nn.functional import cross_entropy,softmax
from transformers import BertModel, BertConfig, AutoConfig, AutoModelForSequenceClassification
import copy
from accelerate.utils import set_seed
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
set_seed(42)#for generating scanpaths

class Joint_Gaze_LM(nn.Module):
	def __init__(self, sp_gen_model, gaze_LM, args):
		super(Joint_Gaze_LM, self).__init__()
		self.sp_gen_model = sp_gen_model
		self.gaze_LM = gaze_LM
		self.args = args

	def convert_word_pos_seq_to_token_pos_seq(self, word_pos_1, sn_len_1,
													word_pos_2, sn_len_2,
													word_ids_sn
													):
		#Find the number "sn_len+1" -> the end point
		stop_mask_1 = (word_pos_1 == (sn_len_1+1).unsqueeze(1))
		stop_mask_1 = ~(stop_mask_1.cumsum(dim=1).cumsum(dim=1) == 1).cumsum(dim=1).bool()

		if word_pos_2 is not None:
			stop_mask_2 = (word_pos_2 == (sn_len_2+1).unsqueeze(1))
			stop_mask_2 = ~(stop_mask_2.cumsum(dim=1).cumsum(dim=1) == 1).cumsum(dim=1).bool()

		#compute gaze token position
		token_ids_sn = torch.arange(word_ids_sn.shape[1]).unsqueeze(0).expand(word_ids_sn.shape[0],-1).to(word_pos_1.device)
		word_ids_2_token_ids_sn = token_ids_sn - word_ids_sn

		gaze_token_pos = []
		for b in range(word_pos_1.shape[0]):
			#remove invalid predictions + SEP token
			valid_pos_seq = torch.masked_select(word_pos_1[b,:], stop_mask_1[b,:])

			if word_pos_2 is not None:
				SEP_indx = sn_len_1[b].reshape(1)+1
				valid_pos_seq = torch.cat((valid_pos_seq, SEP_indx)) #add SEP token back to differentiate two sentences

				valid_pos_seq_2 = torch.masked_select(word_pos_2[b,:], stop_mask_2[b,:])[1:] + SEP_indx
				valid_pos_seq = torch.cat((valid_pos_seq, valid_pos_seq_2))

			#remove CLS token
			valid_pos_seq = valid_pos_seq[1:]

			#convert word pos sequence to token pos sequence
			cur_token_pos = torch.tensor([0], dtype=torch.float64).to(word_pos_1.device) # fake CLS token for tensor concatenation
			for p in valid_pos_seq:
				idx = torch.where(word_ids_sn[b]==p)[0]
				for i in idx:
					cur_token_pos = torch.cat((cur_token_pos, (p + word_ids_2_token_ids_sn[b][i]).reshape(1)))
			gaze_token_pos.append(cur_token_pos[1:]) # remove the fake CLS token

		sp_len = [pos.shape[0] for pos in gaze_token_pos]
		sp_len = torch.FloatTensor(sp_len).to(word_pos_1.device)

		#for zero length scanpath, add additional CLS token to avoid error in pack_padded_sequence operation
		for indx in torch.where(sp_len==0)[0]:
			gaze_token_pos[indx] = torch.cat((torch.zeros(1, dtype=torch.float64).to(word_pos_1.device), gaze_token_pos[indx]))
			sp_len[indx] = 1

		# padding. pad first seq to desired length, padding value: 511, last token index that can be retrive from BERT feature layer
		gaze_token_pos[0] = nn.ConstantPad1d((0, 512 - gaze_token_pos[0].shape[0]), 511)(gaze_token_pos[0])
		gaze_token_pos = pad_sequence(gaze_token_pos, batch_first=True, padding_value=511)
		gaze_token_pos = gaze_token_pos[:,:512]
		sp_len[sp_len>512] = 512

		return gaze_token_pos



	def forward(self, batch, le, task_keys):
		sentence1_key, sentence2_key = task_keys
		gaze_pos2=None
		sn_len2=None

		gaze_pos, sn_len = self.sp_gen_model(sn_emd = batch[f'{sentence1_key}_input_ids'],
												sn_mask = batch[f'{sentence1_key}_attention_mask'],
												word_ids_sn = batch[f'{sentence1_key}_word_ids'],
												sn_word_len = batch[f'{sentence1_key}_word_len'],
												le = le)

		if sentence2_key is not None:#two sentences
			gaze_pos2, sn_len2 = self.sp_gen_model(sn_emd = batch[f'{sentence2_key}_input_ids'],
													sn_mask = batch[f'{sentence2_key}_attention_mask'],
													word_ids_sn = batch[f'{sentence2_key}_word_ids'],
													sn_word_len = batch[f'{sentence2_key}_word_len'],
													le = le)

		gaze_token_pos = self.convert_word_pos_seq_to_token_pos_seq(word_pos_1=gaze_pos,
																	sn_len_1=sn_len,
																	word_pos_2=gaze_pos2,
																	sn_len_2=sn_len2,
																	word_ids_sn=batch[f'NLP_model_word_ids'])


		out = self.gaze_LM(batch, gaze_token_pos)
		return out

class PLM_AS(nn.Module):
	def __init__(self, args, seed):
		super(PLM_AS, self).__init__()
		set_seed(seed)
		# Load pretrained model and tokenizer
		# download model & vocab.
		config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels, finetuning_task=args.task_name)
		config.output_hidden_states=True
		init_model = AutoModelForSequenceClassification.from_pretrained(
			args.model_name_or_path,
			from_tf=bool(".ckpt" in args.model_name_or_path),
			config=config,
			ignore_mismatched_sizes=False,
		)

		self.model = init_model
		#freeze the classification layer parameters in Bert model
		for param in self.model.bert.pooler.parameters():
			param.requires_grad = False
		for param in self.model.classifier.parameters():
			param.requires_grad = False

		#add new layers for processing scanpath and do final prediction
		self.dropout = nn.Dropout(0.1)
		self.gru = nn.GRU(input_size=768,
							hidden_size=768,
							num_layers=1,
							batch_first=True,
							bidirectional=False)


		self.classifier = nn.Linear(768, args.num_labels)

	def forward(self, batch, gaze_pos):

		x = self.model(input_ids=batch['NLP_model_input_ids'],
						attention_mask=batch['NLP_model_attention_mask'],
						token_type_ids=batch['NLP_model_token_type_ids'])
		x = x.hidden_states[-1]

		#retrieve features according to scanpath ordering, this is non-differentiable operation during training
		#x_sp = torch.gather(x, 1, gaze_pos.unsqueeze(2).repeat(1,1,768))
		#instead
		#make own one-hot encoding so that it is differentiable during training
		token_ids_sn = torch.arange(x.shape[1])[None, None, :].expand(gaze_pos.shape[0], gaze_pos.shape[1], -1).to(gaze_pos.device)
		one_hot = token_ids_sn - gaze_pos.unsqueeze(-1)
		one_hot[one_hot!=0] = 1
		one_hot = 1 - one_hot

		x_sp = torch.einsum('bij,bki->bkj', x, one_hot.float())
		x_sp = self.dropout(x_sp)

		x_sp_len = (gaze_pos!=511).sum(1)
		x_sp_packed = pack_padded_sequence(x_sp, x_sp_len.cpu(), batch_first=True, enforce_sorted=False)
		x_sp_packed, last_hidden = self.gru(x_sp_packed, x[:,0,:].unsqueeze(0).contiguous())
		#output_padded, output_lengths = pad_packed_sequence(x_sp_packed, batch_first=True)

		out = self.classifier(last_hidden)

		return out

class Eyettention(nn.Module):
	def __init__(self, cf):
		super(Eyettention, self).__init__()
		self.cf = cf
		self.window_width = 1
		self.hidden_size = 128

		#encoder
		encoder_config = BertConfig.from_pretrained(self.cf["model_pretrained"])
		encoder_config.output_hidden_states=True
		 # initiate Bert with pre-trained weights
		print("keeping Bert with pre-trained weights")
		self.bert = BertModel.from_pretrained(self.cf["model_pretrained"], config = encoder_config)
		self.bert.eval()
		#freeze the parameters in Bert model
		for param in self.bert.parameters():
			param.requires_grad = False

		self.embedding_dropout = nn.Dropout(0.4)
		self.encoder_lstm1 = nn.LSTM(input_size = 768, hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm2 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm3 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm4 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm5 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm6 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm7 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm8 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)

		#decoder
		self.position_embeddings = nn.Embedding(encoder_config.max_position_embeddings, encoder_config.hidden_size)
		self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)
		self.attn_position = nn.Linear(self.hidden_size, self.hidden_size+1) #acoount for the word length feature

		#initialize eight decoder cells
		self.decoder_cell1 = nn.LSTMCell(768, self.hidden_size)
		self.decoder_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell4 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell5 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell6 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell7 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell8 = nn.LSTMCell(self.hidden_size, self.hidden_size)

		#fixation postion decoder
		self.decoder_dense1 = nn.Linear(self.hidden_size*2+1, 512)
		self.decoder_dense2 = nn.Linear(512, 256)
		self.decoder_dense3 = nn.Linear(256, 256)
		self.decoder_dense4 = nn.Linear(256, 256)
		#initialize last dense layer
		self.decoder_dense5 = nn.Linear(256, cf["used_sn_len"]*2-3)
		self.dropout_LSTM = nn.Dropout(0.2)
		self.dropout_dense = nn.Dropout(0.2)

		#for scanpath generation
		self.softmax = nn.Softmax(dim=1)

	def pool_subwords_to_word(self, subword_emb, word_ids_sn, target, pool_method='sum'):
		#try batching computing
		# Pool bert subwords back to word level
		merged_word_att = torch.empty(subword_emb.shape[0], 0, 768).to(subword_emb.device)
		if target == 'sn':
			max_len = self.cf["max_sn_len"] #CLS and SEP included
		elif target == 'sp':
			max_len = self.cf["max_sp_len"] - 1 #do not account the 'SEP' token

		for word_idx in range(max_len):
			word_mask = (word_ids_sn == word_idx).unsqueeze(2).repeat(1, 1, 768)
			#pooling method -> sum
			if pool_method=='sum':
				pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			elif pool_method=='mean':
				pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			merged_word_att = torch.cat([merged_word_att, pooled_word_emb], dim=1)
		mask_word = torch.sum(merged_word_att, 2).bool()
		return merged_word_att, mask_word


	def encode(self, sn_emd, sn_mask, word_ids_sn, sn_word_len):
		outputs = self.bert(input_ids=sn_emd, attention_mask=sn_mask)
		hidden_rep_orig, pooled_rep = outputs[0], outputs[1]
		# Pool bert subwords back to word level for english corpus
		merged_word_att, sn_mask_word = self.pool_subwords_to_word(hidden_rep_orig,
																	word_ids_sn,
																	target='sn',
																	pool_method='sum')

		hidden_rep = self.embedding_dropout(merged_word_att)
		#eight LSTM layers for encoder
		x, (hn, hc) = self.encoder_lstm1(hidden_rep, None)
		x, (hn, hc) = self.encoder_lstm2(self.dropout_LSTM(x), None)
		residual = x
		x, (hn, hc) = self.encoder_lstm3(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm4(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm5(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm6(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm7(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm8(self.dropout_LSTM(x), None)
		x = x + residual

		#concatenate with the word length feature
		x = torch.cat((x, sn_word_len[:, :, None]), dim=2)
		return x, sn_mask_word

	def location_prediction(self, sp_enc_out, word_enc_out, sp_pos, sn_mask, timestep):
		#predict fixation location
		# General Attention:
		# score(ht,hs) = (ht^T)(Wa)hs
		# hs is the output from encoder
		# ht is the previous hidden state from decoder
		# self.attn(o): [batch, step, units]
		attn_prod = torch.matmul(self.attn_position(sp_enc_out.unsqueeze(1)), word_enc_out.permute(0,2,1)) # [batch, 1, step]
		#local attention
		aligned_position = sp_pos[:, timestep]

		# Get window borders
		left = torch.where(aligned_position - self.window_width >= 0, (aligned_position - self.window_width), torch.tensor(0, dtype=torch.float).to(sn_mask.device))
		right = torch.where(aligned_position + self.window_width <= self.cf["max_sn_len"]-1, aligned_position + self.window_width, torch.tensor(self.cf["max_sn_len"]-1, dtype=torch.float).to(sn_mask.device))


		#exclude padding tokens
		#only consider words in the window
		sen_seq = torch.arange(self.cf["max_sn_len"])[None,:].expand(sn_mask.shape[0],self.cf["max_sn_len"]).to(sn_mask.device)
		outside_win_mask = (sen_seq < left.unsqueeze(1)) +  (sen_seq > right.unsqueeze(1))
		attn_prod += (~sn_mask + outside_win_mask).unsqueeze(1) * -1e9
		att_weight = softmax(attn_prod, dim=2)             # [batch, 1, step]

		#atten_weights_batch = torch.cat([atten_weights_batch, att_weight], dim=1)
		context = torch.matmul(att_weight, word_enc_out)    # [batch, 1, units]
		hc = torch.cat([context.squeeze(1),sp_enc_out],dim=1)      # [batch, units *2]

		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense1(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense2(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense3(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense4(hc))
		result = self.decoder_dense5(hc)                   # [batch, dec_o_dim]
		return result

	def decode(self, sn_mask, word_enc_out, sn_emd, word_ids_sn, le):
		sn_len = (torch.sum(sn_mask, axis=1)-2).float()
		# Initialize hidden state and cell state with zeros,
		hn = torch.zeros(8, sn_mask.shape[0], self.hidden_size).to(sn_mask.device)
		hc = torch.zeros(8, sn_mask.shape[0], self.hidden_size).to(sn_mask.device)
		hx, cx = hn[0,:,:], hc[0,:,:]
		hx2, cx2 = hn[1,:,:], hc[1,:,:]
		hx3, cx3 = hn[2,:,:], hc[2,:,:]
		hx4, cx4 = hn[3,:,:], hc[3,:,:]
		hx5, cx5 = hn[4,:,:], hc[4,:,:]
		hx6, cx6 = hn[5,:,:], hc[5,:,:]
		hx7, cx7 = hn[6,:,:], hc[6,:,:]
		hx8, cx8 = hn[7,:,:], hc[7,:,:]

		#use CLS token (101) as start token
		dec_in_start = (torch.ones(sn_mask.shape[0]) * 101).long().to(sn_mask.device)
		dec_emb_in = self.bert.embeddings.word_embeddings(dec_in_start) # [batch, emb_dim]

		#add positional embeddings
		start_pos = torch.zeros(sn_mask.shape[0]).to(sn_mask.device)
		position_embeddings = self.position_embeddings(start_pos.long())
		dec_emb_in = dec_emb_in+position_embeddings
		dec_emb_in = self.LayerNorm(dec_emb_in)
		dec_in = self.embedding_dropout(dec_emb_in)

		#generate fixation one by one in an autoregressive way
		output_pos = torch.empty(sn_mask.shape[0], 0, requires_grad=True).to(sn_mask.device)
		pred_counter = 0
		output_pos = torch.cat([output_pos, start_pos.unsqueeze(1)], dim=1)
		for p in range(self.cf['max_pred_len']-1):
			hx, cx = self.decoder_cell1(dec_in, (hx, cx))     # [batch, units]
			hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
			residual = hx2
			hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
			input3 = hx3 + residual
			residual = input3
			hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(input3), (hx4, cx4))
			input4 = hx4 + residual
			residual = input4
			hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(input4), (hx5, cx5))
			input5 = hx5 + residual
			residual = input5
			hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(input5), (hx6, cx6))
			input6 = hx6 + residual
			residual = input6
			hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(input6), (hx7, cx7))
			input7 = hx7 + residual
			residual = input7
			hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(input7), (hx8, cx8))
			input8 = hx8 + residual

			#location prediction
			pred_loc_logits = self.location_prediction(input8, word_enc_out, output_pos, sn_mask, p)
			if self.training:
				#Sample hard categorical using "Straight-through" trick:
				sampled_pred_loc = F.gumbel_softmax(pred_loc_logits, tau=self.cf["tau"], hard=True)
			else:
				#sampling next fixation location according to the distribution
				sampled_pred_loc = torch.multinomial(self.softmax(pred_loc_logits), 1).squeeze()
				#sampled_pred_loc = pred_loc_logits.argmax(1)
				sampled_pred_loc = F.one_hot(sampled_pred_loc, num_classes=le.classes_.shape[0])

			#print(sampled_pred_loc.grad_fn)
			sac_length_class = torch.tensor(le.classes_).to(sn_mask.device).repeat(sn_mask.shape[0],1)
			sampled_sac_length = (sac_length_class * sampled_pred_loc).sum(1)
			#add saccade length -> predicted fixation word index
			pred_word_index = (output_pos[:, -1] + sampled_sac_length)

			#check the output word index for validity
			#when the prediction is end-of-sentence (23) -- set to sentence length+1, i.e. token <'SEP'>
			pred_word_index[sampled_sac_length == 23] = sn_len[sampled_sac_length == 23]+1
			#when the predicted fixation word index larger than sentence max length -- set to sentence length+1, i.e. token <'SEP'>
			pred_word_index[pred_word_index > sn_len] = sn_len[pred_word_index > sn_len]+1
			#predicted fixation word index smaller than 1 -- set to 1
			pred_word_index[pred_word_index < 1] = 1
			output_pos = torch.cat([output_pos, pred_word_index.unsqueeze(1)], dim=1)

			#prepare next timestamp input token
			pred_counter += 1
			#use predictions (token ids) as input to the next timestep
			input_ids = sn_emd * (word_ids_sn == pred_word_index.unsqueeze(1))
			mask_input_ids = ~(input_ids==0).unsqueeze(2).repeat(1,1,768)
			#merge tokens
			dec_emb_in = torch.sum(self.bert.embeddings.word_embeddings(input_ids) * mask_input_ids, axis=1)

			#add positional embeddings
			position_embeddings = self.position_embeddings(output_pos[:, -1].long())
			dec_emb_in = dec_emb_in+position_embeddings
			dec_emb_in = self.LayerNorm(dec_emb_in)
			dec_emb_in = self.embedding_dropout(dec_emb_in)

		return output_pos, sn_len                         # [batch, step, dec_o_dim]


	def forward(self, sn_emd, sn_mask, word_ids_sn, sn_word_len, le):
		x, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len)                  # [batch, step, units], [batch, units]
		pred_pos, sn_len = self.decode(sn_mask_word, x, sn_emd, word_ids_sn, le)    # [batch, step, dec_o_dim]
		return pred_pos, sn_len
