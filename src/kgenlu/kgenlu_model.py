import torch
from tqdm import tqdm
from kgenlu_read_data import prepare_data, Sample, domain_slot_list, domain_slot_type_map
from kgenlu_config import args, logger, PAD_token
from torch import nn
from transformers import RobertaModel, AlbertModel

no_value_assign_strategy = args['no_value_assign_strategy']


def unit_test():
    pretrained_model = args['pretrained_model']
    name = args['name']
    data, classify_slot_value_index_dict, classify_slot_index_value_dict = prepare_data(overwrite=False)
    model = KGENLU(name, pretrained_model, classify_slot_value_index_dict)
    train_loader, dev_loader, test_loader = data
    for batch_data in tqdm(train_loader):
        model(batch_data)
    logger.info('feed success')


class KGENLU(nn.Module):
    def __init__(self, name, pretrained_model, classify_slot_value_index_dict):
        super(KGENLU, self).__init__()
        self.name = name
        self.embedding_dim = args['encoder_d_model']
        self.encoder = PretrainedEncoder(pretrained_model)
        self.classify_slot_value_index_dict = classify_slot_value_index_dict
        # Gate dict
        self.gate, self.slot_parameter, self.referred_parameter = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        self.slot_initialize()

    def slot_initialize(self):
        """
        Initialize slot relevant parameters and index mappings
        """
        for domain_slot in domain_slot_type_map:
            # none, dont care, inform, referred, hit
            self.gate[domain_slot] = nn.Linear(self.embedding_dim, 5)
            if no_value_assign_strategy == 'miss':
                self.referred_parameter[domain_slot] = nn.Linear(self.embedding_dim, 30)
            else:
                self.referred_parameter[domain_slot] = nn.Linear(self.embedding_dim, 31)
            if domain_slot_type_map[domain_slot] == 'classify':
                if no_value_assign_strategy == 'miss':
                    num_value = len(self.classify_slot_value_index_dict[domain_slot])
                else:
                    # if not hit, the target index is len(classify_slot_value_index_dict[domain_slot]) rather
                    # than -1
                    num_value = len(self.classify_slot_value_index_dict[domain_slot]) + 1
                self.slot_parameter[domain_slot] = nn.Linear(self.embedding_dim, num_value)
            elif domain_slot_type_map[domain_slot] == 'span':
                # probability of start index and end index
                self.slot_parameter[domain_slot] = nn.Linear(self.embedding_dim, 2)
            else:
                raise ValueError('Error Value')

    def forward(self, data):
        """
        context token id shape [batch size, sequence length]
        """
        active_domain = data[1]
        active_slot = data[2]
        context_token = data[5]
        context_mask = (1 - data[9].type(torch.uint8))
        # predict
        referred_dict, hit_type_dict, hit_value_dict = {}, {}, {}
        for domain_slot in domain_slot_list:
            referred_dict[domain_slot] = data[6][domain_slot]
            hit_type_dict[domain_slot] = data[7][domain_slot]
            hit_value_dict[domain_slot] = data[8][domain_slot]

        encode = self.encoder(context_token, padding_mask=context_mask)
        predict_gate, predict_dict, referred_dict = {}, {}, {}
        # Choose the output of the first token ([CLS]) to predict gate and classification)
        for domain_slot in domain_slot_list:
            predict_gate[domain_slot] = self.gate[domain_slot](encode[:, 0, :])
            referred_dict[domain_slot] = self.referred_parameter[domain_slot](encode[:, 0, :])
            slot_type, weight = domain_slot_type_map[domain_slot], self.slot_parameter[domain_slot]
            if slot_type == 'classify':
                predict_dict[domain_slot] = weight(encode[:, 0, :])
            else:
                predict_dict[domain_slot] = weight(encode)
        return predict_gate, predict_dict, referred_dict

    @staticmethod
    def create_mask(inputs):
        sequence_length = inputs.shape[0]
        input_mask = torch.zeros((sequence_length, sequence_length)).type(torch.bool)
        input_padding_mask = inputs == PAD_token
        return input_mask, input_padding_mask


class PretrainedEncoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super(PretrainedEncoder, self).__init__()
        self._model_name = pretrained_model_name
        if pretrained_model_name == 'roberta':
            self.model = RobertaModel.from_pretrained('roberta-base')
        elif pretrained_model_name == 'albert':
            self.model = AlbertModel.from_pretrained('albert-base-v2')
        else:
            ValueError('Invalid Pretrained Model')

    def forward(self, context, padding_mask):
        """
        :param context: [sequence_length, batch_size]
        :param padding_mask: [sequence_length, batch_size]
        :return: output:  [sequence_length, batch_size, word embedding]
        """
        # required format: [batch_size, sequence_length]
        if self._model_name == 'roberta':
            assert context.shape[1] <= 512
        if self._model_name == 'roberta':
            output = self.model(context, attention_mask=padding_mask)['last_hidden_state']
            return output
        if self._model_name == 'albert':
            output = self.model(context, attention_mask=padding_mask)['last_hidden_state']
            return output
        else:
            ValueError('Invalid Pretrained Model')


if __name__ == '__main__':
    unit_test()

