import os
import math
import torch
from tqdm import tqdm
from torch import nn, Tensor
from torch.nn.functional import softmax
from evaluation import evaluation_batch, comprehensive_evaluation
from transformers import RobertaModel, AlbertModel
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from transformers import get_linear_schedule_with_warmup
from multiwoz_util import prepare_data
from multiwoz_config import args, DEVICE, PAD_token, logger
import pickle
from torch import optim
from datetime import datetime

max_seq_len = args['max_sentence_length']
multi_GPU_flag = args['multi_gpu']
if multi_GPU_flag:
    logger.info("Using Multiple GPU (if possible) to optimize model")
    logger.info('number of available GPU: {}'.format(torch.cuda.device_count()))
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args['local_rank'])


class Encoder(nn.Module):
    """
    Transformer based utterance & history encoder
    """
    def __init__(self, vocab_size, hidden_size, dropout, d_model, n_head, activation, num_encoder_layers,
                 dim_feed_forward):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout).to(DEVICE)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, n_head, dim_feed_forward, dropout, activation),
            num_encoder_layers,
            LayerNorm(d_model)
        ).to(DEVICE)

        # load pretrained word embedding
        self.embedding = nn.Embedding(self.vocab_size, self.d_model).to(DEVICE)
        if args["load_embedding"]:
            e = pickle.load(open(
                os.path.join(args['aligned_embedding_path'], 'glove_42B_embed_{}'.format(self.vocab_size)), 'rb'))
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(e))
            logger.info('Pretrained Embedding Loaded')
        else:
            nn.init.xavier_normal_(self.embedding.weight)
            logger.info('Training Embedding from scratch')
        if args["update_embedding"]:
            self.embedding.weight.requires_grad = True
            logger.info('Update Embedding in training phase')
        else:
            self.embedding.weight.requires_grad = False
            logger.info('Do not update Embedding in training phase')

    def forward(self, context, padding_mask):
        padding_mask = padding_mask.transpose(0, 1)
        context = self.embedding(context) * math.sqrt(self.d_model)
        encode = self.pos_encoder(context)
        encode = self.encoder(encode, src_key_padding_mask=padding_mask)
        return encode


def train_process(dataset, model, optimizer, cross_entropy, epoch, scheduler):
    dataset = tqdm(enumerate(dataset), total=len(dataset))
    full_loss = 0
    for i, batch in dataset:
        optimizer.zero_grad()
        predict_gate_logits, predict_dict_logits = model(batch)
        batch_loss = 0
        for slot_name in predict_gate_logits:
            gate_slot_predict = predict_gate_logits[slot_name]
            gate_slot_label = torch.tensor(batch['gate'][slot_name], dtype=torch.long).to(DEVICE)
            value_slot_predict = predict_dict_logits[slot_name]
            value_slot_label = torch.tensor(batch['label'][slot_name], dtype=torch.long).to(DEVICE)

            gate_loss = cross_entropy(gate_slot_predict, gate_slot_label) * (1-args['span_loss_ratio'])
            if batch['slot_type_dict'][0][slot_name] == 'classify':
                value_loss = cross_entropy(value_slot_predict, value_slot_label) * (1-args['span_loss_ratio'])
            else:
                # for long truncate case, the interested token is beyond the maximum sentence length
                value_slot_label = torch.where(value_slot_label < max_seq_len, value_slot_label, -1)
                value_loss = args['span_loss_ratio'] * \
                    (cross_entropy(value_slot_predict[:, :, 0].transpose(1, 0), value_slot_label[:, 0]) +
                     cross_entropy(value_slot_predict[:, :, 1].transpose(1, 0), value_slot_label[:, 1]))
            if args['multi_gpu']:
                batch_loss += value_loss.mean() + gate_loss.mean()
            else:
                batch_loss += value_loss + gate_loss
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        full_loss += batch_loss
    logger.info('epoch: {}, average_loss: {}'.format(epoch, full_loss/dataset.total))


def validation_and_test_process(dataset, model, slot_info_dict):
    predicted_result = []
    dataset = tqdm(enumerate(dataset), total=len(dataset))
    for i, batch in dataset:
        predict_gate_logits, predict_logits_dict = model(batch)
        # normalized probability
        predict_gate, predict_dict = {}, {}
        for slot_name in predict_gate_logits:
            predict_gate[slot_name] = softmax(predict_gate_logits[slot_name], dim=1)
            if slot_info_dict['slot_type_dict'][slot_name] == 'classify':
                predict_dict[slot_name] = softmax(predict_logits_dict[slot_name], dim=1)
            else:
                predict_dict[slot_name] = softmax(predict_logits_dict[slot_name], dim=0)
        predicted_result.append(evaluation_batch(predict_gate, predict_dict, batch, slot_info_dict))
    return predicted_result



