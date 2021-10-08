import os
from tqdm import tqdm
from torch import nn
from torch.nn.functional import softmax
from evaluation import evaluation_batch, comprehensive_evaluation
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from multiwoz_util import PAD_token, prepare_data, load_graph_embeddings, graph_embeddings_alignment
from multiwoz_config import args, DEVICE
import pickle


class KGENLU(nn.Module):
    def __init__(self, word_index_dict, slot_info_dict, name='KGENLU'):
        super(KGENLU, self).__init__()
        self.name = name
        self.vocab_size = len(word_index_dict)
        self.embedding_dim = args['encoder_d_model']
        self.word_index_dict = word_index_dict
        self.encoder = Encoder(
            vocab_size=self.vocab_size,
            hidden_size=args['encoder_hidden_size'],
            dropout=args['encoder_dropout'],
            d_model=self.embedding_dim,
            n_head=args['encoder_n_head'],
            activation=args['encoder_activation'],
            num_encoder_layers=args['encoder_num_encoder_layers'],
            dim_feed_forward=args['encoder_dim_feed_forward']
        )

        # Gate dict
        # 3 means None, Don't Care, Hit
        self.gate, self.slot_type_dict, self.classify_slot_index_value_dict, self.slot_parameter = {}, {}, {}, {}
        self.slot_initialize(slot_info_dict)

        # load pretrained word embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD_token).to(DEVICE)
        if args["load_embedding"]:
            e = pickle.load(open(os.path.join(args['aligned_embedding_path'],
                                              'glove_42B_embed_{}'.format(self.vocab_size)), 'rb'))
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(e))
            print('Pretrained Embedding Loaded')
        else:
            nn.init.xavier_normal(self.embedding)
            print('Training Embedding from scratch')
        if args["update_embedding"]:
            self.embedding.weight.requires_grad = True
            print('Update Embedding in training phase')
        else:
            self.embedding.weight.requires_grad = False
            print('Do not update Embedding in training phase')

        # load graph embedding
        wordnet_embedding_path = args['wordnet_embedding_path']
        wordnet_path = args['wordnet_path']
        entity_embed_dict, relation_embed_dict = load_graph_embeddings(wordnet_embedding_path, wordnet_path)
        self.aligned_graph_embedding = graph_embeddings_alignment(entity_embed_dict, word_index_dict)
        self.full_graph_entity_embed_dict = entity_embed_dict
        self.full_relation_embed_dict = relation_embed_dict

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
        context_embedding = self.embedding(context)
        encode = self.encoder(context_embedding)

        predict_gate = {}
        predict_dict = {}
        for slot_name in label:
            predict_gate[slot_name] = softmax(self.gate[slot_name](encode[0, :, :]), dim=1)
            slot_type = self.slot_type_dict[slot_name]
            weight = self.slot_parameter[slot_name]
            if slot_type == 'classify':
                predict_dict[slot_name] = softmax(weight(encode[0, :, :]), dim=1)
            else:
                predict_dict[slot_name] = softmax(weight(encode), dim=0)
        return predict_gate, predict_dict


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

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, n_head, dim_feed_forward, dropout, activation),
            num_encoder_layers,
            LayerNorm(d_model)
        ).to(DEVICE)

    def forward(self, context):
        encode = self.encoder(context)
        return encode


def train_process(dataset, model):
    dataset = tqdm(enumerate(dataset), total=len(dataset))
    for i, batch in dataset:
        model(batch)


def validation_and_test_process(dataset, model, slot_info_dict):
    predicted_result = []
    dataset = tqdm(enumerate(dataset), total=len(dataset))
    for i, batch in dataset:
        predict_gate, predict_dict = model(batch)
        predicted_result.append(evaluation_batch(predict_gate, predict_dict, batch, slot_info_dict))
    return predicted_result


def main():
    file_path = args['cache_path']
    train, val, test, word_index_stat, slot_value_dict, slot_type_dict =\
        prepare_data(read_from_cache=True, file_path=file_path)
    slot_info_dict = {'slot_value_dict': slot_value_dict, 'slot_type_dict': slot_type_dict}
    model = KGENLU(word_index_stat.word2index, slot_info_dict)

    for epoch in range(2):
        print("Epoch :{}".format(epoch))
        # Run the train function
        train_process(train, model)
        validation_result = validation_and_test_process(val, model, slot_info_dict)
        test_result = validation_and_test_process(test, model, slot_info_dict)
        comprehensive_evaluation(validation_result, slot_info_dict, 'validation')
        comprehensive_evaluation(test_result, slot_info_dict, 'test')


if __name__ == '__main__':
    main()
