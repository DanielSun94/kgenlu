import os
import math
import torch
from tqdm import tqdm
from torch import nn, Tensor
from torch.nn.functional import softmax
from evaluation import evaluation_batch, comprehensive_evaluation
from transformers import RobertaModel, AlbertModel
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from multiwoz_util import prepare_data
from multiwoz_config import args, DEVICE, PAD_token
import pickle
from torch import optim
import logging

max_seq_len = args['max_sentence_length']

class KGENLU(nn.Module):
    def __init__(self, word_index_stat, slot_info_dict, name, pretrained_model):
        super(KGENLU, self).__init__()
        self.name = name
        self.embedding_dim = args['encoder_d_model']
        if pretrained_model is None:
            self.word_index_dict = word_index_stat.word2index
            self.encoder = CustomEncoder(
                vocab_size=len(word_index_stat.word2index),
                hidden_size=args['encoder_hidden_size'],
                dropout=args['encoder_dropout'],
                d_model=self.embedding_dim,
                n_head=args['encoder_n_head'],
                activation=args['encoder_activation'],
                num_encoder_layers=args['encoder_num_encoder_layers'],
                dim_feed_forward=args['encoder_dim_feed_forward']
            )
        else:
            self.encoder = PretrainedEncoder(pretrained_model)

        # Gate dict
        # 3 means None, Don't Care, Hit
        self.gate, self.slot_type_dict, self.classify_slot_index_value_dict, self.slot_parameter = \
                nn.ModuleDict(), {}, {}, nn.ModuleDict()
        self.slot_initialize(slot_info_dict)

        # load graph embedding
        # wordnet_embedding_path = args['wordnet_embedding_path']
        # wordnet_path = args['wordnet_path']
        # entity_embed_dict, relation_embed_dict = load_graph_embeddings(wordnet_embedding_path, wordnet_path)
        # self.aligned_graph_embedding = graph_embeddings_alignment(entity_embed_dict, word_index_dict)
        # self.full_graph_entity_embed_dict = entity_embed_dict
        # self.full_relation_embed_dict = relation_embed_dict

    def slot_initialize(self, slot_info_dict):
        """
        Initialize slot relevant parameters and index mappings
        """
        slot_value_dict, slot_type_dict = slot_info_dict['slot_value_dict'], slot_info_dict['slot_type_dict']
        self.slot_type_dict = slot_type_dict
        for slot_name in slot_type_dict:
            self.gate[slot_name] = nn.Linear(self.embedding_dim, 3).to(DEVICE)
            if slot_type_dict[slot_name] == 'classify':
                assert len(slot_value_dict[slot_name]) < args['span_limit']
                self.slot_parameter[slot_name] = \
                    nn.Linear(self.embedding_dim, len(slot_value_dict[slot_name])).to(DEVICE)
                self.classify_slot_index_value_dict[slot_name] = {}
                for index, value in enumerate(slot_value_dict[slot_name]):
                    self.classify_slot_index_value_dict[slot_name][index] = value
            elif slot_type_dict[slot_name] == 'span':
                # probability of start index and end index
                self.slot_parameter[slot_name] = nn.Linear(self.embedding_dim, 2).to(DEVICE)
            else:
                raise ValueError('Error Value')

    def forward(self, data):
        """
        context shape [sentence_length, batch_size, token_embedding]
        """
        label = data['label']
        context = data['context'].transpose(1, 0).to(DEVICE)
        input_mask, input_padding_mask = self.create_mask(context)
        encode = self.encoder(context, padding_mask=input_padding_mask)

        predict_gate = {}
        predict_dict = {}

        # Choose the output of the first token ([CLS]) to predict gate and classification)
        for slot_name in label:
            predict_gate[slot_name] = self.gate[slot_name](encode[0, :, :])
            slot_type, weight = self.slot_type_dict[slot_name], self.slot_parameter[slot_name]
            if slot_type == 'classify':
                predict_dict[slot_name] = weight(encode[0, :, :])
            else:
                predict_dict[slot_name] = weight(encode)
        return predict_gate, predict_dict

    @staticmethod
    def create_mask(inputs):
        sequence_length = inputs.shape[0]
        input_mask = torch.zeros((sequence_length, sequence_length), device=DEVICE).type(torch.bool)
        input_padding_mask = inputs==PAD_token
        return input_mask, input_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, mex_length: int = 1000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, mex_length).reshape(mex_length, 1)
        pos_embedding = torch.zeros((mex_length, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class CustomEncoder(nn.Module):
    """
    Transformer based utterance & history encoder
    """
    def __init__(self, vocab_size, hidden_size, dropout, d_model, n_head, activation, num_encoder_layers,
                 dim_feed_forward):
        super(CustomEncoder, self).__init__()
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
            print('Pretrained Embedding Loaded')
        else:
            nn.init.xavier_normal_(self.embedding.weight)
            print('Training Embedding from scratch')
        if args["update_embedding"]:
            self.embedding.weight.requires_grad = True
            print('Update Embedding in training phase')
        else:
            self.embedding.weight.requires_grad = False
            print('Do not update Embedding in training phase')

    def forward(self, context, padding_mask):
        padding_mask = padding_mask.transpose(0, 1)
        context = self.embedding(context) * math.sqrt(self.d_model)
        encode = self.pos_encoder(context)
        encode = self.encoder(encode, src_key_padding_mask=padding_mask)
        return encode


def train_process(dataset, model, optimizer, cross_entropy, epoch):
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

            gate_loss = cross_entropy(gate_slot_predict, gate_slot_label)
            if batch['slot_type_dict'][0][slot_name] == 'classify':
                value_loss = cross_entropy(value_slot_predict, value_slot_label)
            else:
                # for long truncate case, the interested token is beyond the maximum sentence length
                value_slot_label = torch.where(value_slot_label < max_seq_len, value_slot_label, -1)
                value_loss = 0.5 * (cross_entropy(value_slot_predict[:, :, 0].transpose(1, 0), value_slot_label[:, 0]) +
                                cross_entropy(value_slot_predict[:, :, 1].transpose(1, 0), value_slot_label[:, 1]))

            batch_loss += value_loss + gate_loss
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'], error_if_nonfinite=False)

        optimizer.step()
        full_loss += batch_loss
    print('epoch: {}, average_loss: {}'.format(epoch, full_loss/dataset.total))


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


class PretrainedEncoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super(PretrainedEncoder, self).__init__()
        self._model_name = pretrained_model_name
        if pretrained_model_name == 'roberta':
            self.model = RobertaModel.from_pretrained('roberta-base').to(DEVICE)
        elif pretrained_model_name == 'albert':
            self.model = AlbertModel.from_pretrained('albert-base-v2').to(DEVICE)
        else:
            ValueError('Invalid Pretrained Model')

    def forward(self, context, padding_mask):
        """
        :param context: [sequence_length, batch_size]
        :param padding_mask: [sequence_length, batch_size]
        :return: output:  [sequence_length, batch_size, word embedding]
        """
        # required format: [batch_size, sequence_length]
        context, padding_mask = context.transpose(0, 1), padding_mask.transpose(0, 1)
        if self._model_name == 'roberta':
            assert context.shape[1] <= 512
        if self._model_name == 'roberta':
            output = self.model(context, attention_mask=padding_mask)['last_hidden_state'].transpose(0, 1)
            return output
        if self._model_name == 'albert':
            output = self.model(context, attention_mask=padding_mask)['last_hidden_state'].transpose(0, 1)
            return output
        else:
            ValueError('Invalid Pretrained Model')


def main():
    file_path = args['cache_path']
    train, val, test, word_index_stat, slot_value_dict, slot_type_dict =\
        prepare_data(read_from_cache=False, file_path=file_path)
    slot_info_dict = {'slot_value_dict': slot_value_dict, 'slot_type_dict': slot_type_dict}
    model = KGENLU(word_index_stat=word_index_stat, slot_info_dict=slot_info_dict,
                   pretrained_model=args['pretrained_model'], name='kgenlu')
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
    learning_rate = args['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    for epoch in range(200):
        print("Epoch :{}".format(epoch))
        # Run the train function
        model.train()
        train_process(train, model, optimizer, cross_entropy, epoch)
        model.eval()

        validation_result = validation_and_test_process(val, model, slot_info_dict)
        comprehensive_evaluation(validation_result, slot_info_dict, 'validation', epoch)
        test_result = validation_and_test_process(test, model, slot_info_dict)
        comprehensive_evaluation(test_result, slot_info_dict, 'test', epoch)
        scheduler.step()


if __name__ == '__main__':
    main()
