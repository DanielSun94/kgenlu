import torch
from torch import mean, cat, stack, LongTensor
import numpy as np
from tqdm import tqdm
from history_read_data import prepare_data, domain_slot_list, domain_slot_type_map, SampleDataset
from history_config import args, logger, DEVICE
from torch.nn import LSTM, ReLU, Linear, Sequential, Module, ModuleDict
from transformers import RobertaModel
from transformers import RobertaTokenizer


if 'roberta' in args['pretrained_model']:
    tokenizer = RobertaTokenizer.from_pretrained(args['pretrained_model'])
else:
    raise ValueError('')
no_value_assign_strategy = args['no_value_assign_strategy']
overwrite_cache = args['overwrite_cache']
lock_embedding_parameter = args['lock_embedding_parameter']
use_multi_gpu = args['multi_gpu']
mentioned_slot_pool_size = args['mentioned_slot_pool_size']


def unit_test():
    pretrained_model = args['pretrained_model']
    name = args['process_name']
    classify_slot_value_index_dict, classify_slot_index_value_dict, train_loader, dev_loader, test_loader = \
        prepare_data(overwrite=overwrite_cache)
    model = HistorySelectionModel(name, pretrained_model, classify_slot_value_index_dict)
    model.train()
    for batch_data in tqdm(train_loader):
        model(batch_data)
    logger.info('feed success')


class HistorySelectionModel(Module):
    def __init__(self, name, pretrained_model, classify_slot_value_index_dict, local_rank=None):
        super(HistorySelectionModel, self).__init__()
        self.name = name
        if use_multi_gpu:
            assert self.local_rank is not None
            self.target_id = local_rank
        else:
            self.target_id = DEVICE
        self.embedding_dim = args['encoder_d_model']
        self.encoder = PretrainedEncoder(pretrained_model)
        self.classify_slot_value_index_dict = classify_slot_value_index_dict
        # Gate dict, 4 for none, dont care, mentioned, hit
        # 暂且不分domain slot specific的参数
        self.gate_predict = ModuleDict()
        self.hit_parameter = ModuleDict()
        for domain_slot in domain_slot_type_map:
            self.gate_predict[domain_slot] = Linear(self.embedding_dim, 4)
            if domain_slot_type_map[domain_slot] == 'classify':
                if no_value_assign_strategy == 'miss':
                    num_value = len(self.classify_slot_value_index_dict[domain_slot])
                else:
                    num_value = len(self.classify_slot_value_index_dict[domain_slot]) + 1
                self.hit_parameter[domain_slot] = Linear(self.embedding_dim, num_value)
            elif domain_slot_type_map[domain_slot] == 'span':
                self.hit_parameter[domain_slot] = Linear(self.embedding_dim, 2)
            else:
                raise ValueError('Error Value')
        # m for mentioned slot
        self.m_query_para = Sequential(Linear(self.embedding_dim, 32), ReLU(), Linear(32, 32), ReLU())
        self.m_slot_para = Sequential(Linear(self.embedding_dim, 32), ReLU(), Linear(32, 32), ReLU())
        self.m_combine_para = Linear(32, 32)

        # 是否锁定embedding的值
        self.token_embedding = self.encoder.model.embeddings.word_embeddings
        if lock_embedding_parameter:
            self.token_embedding.weight.requires_grad = False

        # 由于turn, domain, slot, mentioned type需要经常用到，因此构建一下这些常用策略的embedding
        self.common_token_embedding_dict = None

    def get_common_token_embedding(self):
        common_token_list, common_token_embedding_dict = ['label', 'inform'], {}
        for i in range(0, 30):
            common_token_list.append(str(i))
        for domain_slot in domain_slot_list:
            split_idx = domain_slot.find('-')
            domain, slot = domain_slot[: split_idx], domain_slot[split_idx + 1:].replace('book-', '')
            common_token_list.append(domain)
            common_token_list.append(slot)
        for key in common_token_list:
            token = LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + key))).to(self.target_id)
            common_token_embedding_dict[key] = mean(self.token_embedding(token), dim=0, keepdim=True)
        self.common_token_embedding_dict = common_token_embedding_dict

    def forward(self, data):
        """
        context token id shape [batch size, sequence length]
        """
        id_list, active_domain, active_slot, context_token = data[0], data[1], data[2], data[3]
        context_mask, possible_mentioned_slot_list_dict = (1 - data[4].type(torch.uint8)), data[9]
        possible_mentioned_slot_list_mask_dict = data[10]
        possible_mentioned_slot_list_dict = self.get_mentioned_slots_embedding(possible_mentioned_slot_list_dict)

        encode = self.encoder(context_token, padding_mask=context_mask)
        predict_value_dict = {}
        # Choose the output of the first token ([CLS]) to predict gate and classification)
        # 预测假定hit情况下的预测值
        for domain_slot in domain_slot_list:
            slot_type, weight = domain_slot_type_map[domain_slot], self.hit_parameter[domain_slot]
            if slot_type == 'classify':
                predict_value_dict[domain_slot] = weight(encode[:, 0, :])
            else:
                predict_value_dict[domain_slot] = weight(encode)

        predict_mentioned_slot_dict = self.predict_mentioned_slot_value(
            encode[:, 0, :], possible_mentioned_slot_list_dict, possible_mentioned_slot_list_mask_dict)
        predict_gate_dict = self.predict_gate_value(encode[:, 0, :], possible_mentioned_slot_list_dict,
                                                    possible_mentioned_slot_list_mask_dict)

        return predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict

    def predict_mentioned_slot_value(self, context, mentioned_slots_embedding_dict, mentioned_slot_list_mask_dict):
        predict_mentioned_slot_dict = {}
        for domain_slot in domain_slot_list:
            split_idx = domain_slot.find('-')
            domain, slot = domain_slot[: split_idx], domain_slot[split_idx + 1:].replace('book-', '')
            query = (self.common_token_embedding_dict[domain] + self.common_token_embedding_dict[slot])/2 + context/2
            query_weight = self.m_query_para(query).unsqueeze(dim=2)
            value_weight = self.m_slot_para(mentioned_slots_embedding_dict[domain_slot])
            predicted_value = torch.bmm(value_weight, query_weight).squeeze()
            predicted_value = (~mentioned_slot_list_mask_dict[domain_slot]) * -1e6 + predicted_value
            predict_mentioned_slot_dict[domain_slot] = predicted_value
        return predict_mentioned_slot_dict

    def predict_gate_value(self, context, mentioned_slots_embedding_dict, mentioned_slot_list_mask_dict):
        # 预测Gate值
        gate_predict_dict = {}
        for domain_slot in domain_slot_list:
            slot_embedding_list, mentioned_slots_embedding = [], mentioned_slots_embedding_dict[domain_slot]
            mentioned_slot_mask = mentioned_slot_list_mask_dict[domain_slot].unsqueeze(dim=2)
            valid_length = torch.sum(mentioned_slot_mask, dim=1)
            mentioned_slots_embedding = torch.sum(mentioned_slots_embedding * mentioned_slot_mask, dim=1)
            embedding = mentioned_slots_embedding / valid_length + context
            gate_predict_dict[domain_slot] = self.gate_predict[domain_slot](embedding)
        return gate_predict_dict

    def get_mentioned_slots_embedding(self, possible_mentioned_slot_list_dict):
        # mentioned slots embedding获取，因为数据本身并不齐整(主要是部分value可能一次解析出多个token id)，因此只能这么做
        # TODO 目测这里是速度瓶颈，想想办法能不能快一点
        target_id = self.target_id
        mentioned_slots_embedding_dict = {}
        for domain_slot in domain_slot_list:
            mentioned_slots_embedding_dict[domain_slot] = []
            for sample_idx in range(len(possible_mentioned_slot_list_dict[domain_slot])):
                sample_list = []
                mentioned_slot_list = possible_mentioned_slot_list_dict[domain_slot][sample_idx]
                for mentioned_slot in mentioned_slot_list:
                    turn = mean(self.token_embedding(LongTensor(mentioned_slot[0]).to(target_id)), dim=0, keepdim=True)
                    domain = mean(self.token_embedding(LongTensor(mentioned_slot[1]).to(target_id)), dim=0,
                                  keepdim=True)
                    slot = mean(self.token_embedding(LongTensor(mentioned_slot[2]).to(target_id)), dim=0, keepdim=True)
                    value = mean(self.token_embedding(LongTensor(mentioned_slot[3]).to(target_id)), dim=0, keepdim=True)
                    type_ = mean(self.token_embedding(LongTensor(mentioned_slot[4]).to(target_id)), dim=0, keepdim=True)
                    sample_list.append(mean(cat((value, mean(cat((turn, domain, slot, type_), dim=0), dim=0,
                                                             keepdim=True)), dim=0), dim=0, keepdim=True))
                assert len(sample_list) == mentioned_slot_pool_size
                mentioned_slots_embedding_dict[domain_slot].append(stack(sample_list))
            mentioned_slots_embedding_dict[domain_slot] = stack(mentioned_slots_embedding_dict[domain_slot]).squeeze(2)
        return mentioned_slots_embedding_dict


class PretrainedEncoder(Module):
    def __init__(self, pretrained_model_name):
        super(PretrainedEncoder, self).__init__()
        self._model_name = pretrained_model_name
        if 'roberta' in pretrained_model_name:
            self.model = RobertaModel.from_pretrained(pretrained_model_name)
        else:
            ValueError('Invalid Pretrained Model')

    def forward(self, context, padding_mask):
        """
        :param context: [sequence_length, batch_size]
        :param padding_mask: [sequence_length, batch_size]
        :return: output:  [sequence_length, batch_size, word embedding]
        """
        # required format: [batch_size, sequence_length]
        if 'roberta' in self._model_name:
            assert context.shape[1] <= 512
        if 'roberta' in self._model_name:
            output = self.model(context, attention_mask=padding_mask)['last_hidden_state']
            return output
        else:
            ValueError('Invalid Pretrained Model')


if __name__ == '__main__':
    unit_test()

