import os
from tqdm import tqdm
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from multiwoz_util import PAD_token, prepare_data
from multiwoz_config import args
import pickle


class KGENLU(nn.Module):
    def __init__(self, vocab_size, name='KGENLU'):
        super(KGENLU, self).__init__()
        self.name = name
        self.encoder = Encoder(
            vocab_size=vocab_size,
            hidden_size=args['encoder_hidden_size'],
            dropout=args['encoder_dropout'],
            d_model=args['encoder_d_model'],
            n_head=args['encoder_n_head'],
            activation=args['encoder_activation'],
            num_encoder_layers=args['encoder_num_encoder_layers'],
            dim_feed_forward=args['encoder_dim_feed_forward']
        )

    def forward(self, src):
        self.encoder(src)
        print('TBD')


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
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_token)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, n_head, dim_feed_forward, dropout, activation),
            num_encoder_layers,
            LayerNorm(d_model)
        )

        if args["load_embedding"]:
            e = pickle.load(open(os.path.join(args['aligned_embedding_path'], 'glove_42B_embed_{}'.format(vocab_size)),
                                 'rb'))
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(e))
            print('Pretrained Embedding Loaded')
        else:
            print('Training Embedding from scratch')
        if args["update_embedding"]:
            self.embedding.weight.requires_grad = True
            print('Update Embedding in training phase')
        else:
            self.embedding.weight.requires_grad = False
            print('Do not update Embedding in training phase')

    def forward(self, src):
        print(src)
        print('forward')


def main():
    file_path = os.path.abspath('../resource/multiwoz/cache.pkl')
    train, dev, test, word_index_stat, slot_value_dict, slot_type_dict, slot_value_idx_dict, slot_name_idx_dict = \
        prepare_data(read_from_cache=True, file_path=file_path)
    vocab_size = len(word_index_stat.word2index)
    model = KGENLU(vocab_size=vocab_size)

    for epoch in range(2):
        print("Epoch :{}".format(epoch))
        # Run the train function
        p_bar = tqdm(enumerate(train), total=len(train))
        for i, data in p_bar:
            model(data)


if __name__ == '__main__':
    main()
