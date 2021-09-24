from tqdm import tqdm
from multiwoz_config import args
from multiwoz_util import prepare_data, graph_embeddings_alignment, load_graph_embeddings, load_glove_embeddings
import pickle as pkl
import os
"""
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
"""


def main():
    glove_embedding_path = os.path.abspath('../resource/embedding/glove.42B.300d.txt')
    wordnet_embedding_path = os.path.abspath('../resource/transe_checkpoint/WordNet_2000_checkpoint.tar')
    wordnet_path = os.path.abspath('../resource/wordnet/wordnet_KG.pkl')
    wordnet_obj = pkl.load(open(wordnet_path, 'rb'))
    train, dev, test, word_index_stat, slot_value_dict, slot_type_dict, slot_value_idx_dict, slot_name_idx_dict = \
        prepare_data()
    glove_save_path = os.path.abspath('../resource/embedding/glove_42b_embed_{}'
                                      .format(len(word_index_stat.index2word)))
    glove_embed_mat = load_glove_embeddings(glove_embedding_path, word_index_stat.word2index, glove_save_path)
    entity_embed_dict, relation_embed_dict = load_graph_embeddings(wordnet_embedding_path, wordnet_obj)
    embedding_mat = graph_embeddings_alignment(entity_embed_dict, word_index_stat.word2index)
    train_model(train, dev, test)
    print('util execute accomplished')


def train_model(train, dev, test):
    early_stop = args['early_stop']

    # Configure models and load data
    avg_best, cnt, acc = 0.0, 0, 0.0

    # model = TRADE(hidden_size=int(args['hidden']), lang=lang, path=args['path'], task=args['task'],
    #               lr=float(args['learn']), dropout=float(args['drop']), slots=slots_list, gating_dict=gating_dict)

    # print("[Info] Slots include ", SLOTS_LIST)
    # print("[Info] Unpointable Slots include ", gating_dict)
    print('model restoring process: TBD')

    for epoch in range(2):
        print("Epoch :{}".format(epoch))
        # Run the train function
        p_bar = tqdm(enumerate(train), total=len(train))
        for i, data in p_bar:
            print('model training process: TBD')

        if (epoch + 1) % int(args['eval_epoch']) == 0:
            print('model validation process: TBD')

    print('model testing process: TBD')


if __name__ == '__main__':
    main()
