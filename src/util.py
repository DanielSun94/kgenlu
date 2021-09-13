import numpy as np
import os
import pickle


# preset token tag
UNK, SOS, EOS, PAD = 'UNK', 'SOS', 'EOS', 'PAD'
preset_word_num = 4


def load_glove_embeddings(file_path, word_index_dict, save_path=None):
    """
    load the dict-specific glove embeddings
    The embeddings of out-of-vocab word in the word_index_dict will be assigned as UNK vector
    if the used model has embeddings of UNK, PAD, EOS, SOS. we will use the embeddings in the model, or we will use
    zero vector, average, or random vector.
    """
    if os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    for key in {UNK, SOS, EOS, PAD}:
        assert key in word_index_dict

    print("Loading Glove Model")
    glove_embedding = []
    glove_word_index_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        line_idx = 0
        for line in f:
            split_line = line.split()
            word = split_line[0]
            glove_word_index_dict[word] = line_idx
            word_embedding = np.array(split_line[len(split_line)-300:], dtype=np.float64)
            glove_embedding.append(word_embedding)

            line_idx += 1
    print(f"{len(glove_embedding)} words loaded!")
    glove_embedding = np.array(glove_embedding)

    embedding_dimension = len(glove_embedding[0])
    pad_embedding = np.zeros(embedding_dimension)
    unk_embedding = np.average(glove_embedding, axis=0)
    sos_embedding = (np.random.random(embedding_dimension) - 0.5) * 2
    eos_embedding = (np.random.random(embedding_dimension) - 0.5) * 2

    embedding_mat = np.zeros([len(word_index_dict), embedding_dimension])
    for word in word_index_dict:
        idx = word_index_dict[word]
        if glove_word_index_dict.__contains__(word):
            embedding_mat[idx] = glove_embedding[glove_word_index_dict[word]]
        else:
            if word == PAD:
                embedding_mat[idx] = pad_embedding
            elif word == SOS:
                embedding_mat[idx] = sos_embedding
            elif word == EOS:
                embedding_mat[idx] = eos_embedding
            else:
                embedding_mat[idx] = unk_embedding

    pickle.dump(embedding_mat, open(save_path, 'wb'))
    return embedding_mat


def load_graph_embeddings(file_path):
    pass


def tokenize(string_list, word_index_dict):
    token_list = []
    for word in string_list:
        if word_index_dict.__contains__(word):
            token_list.append(word_index_dict[word])
        else:
            token_list.append(UNK)
    return np.array(token_list)


def embedding(tokenize_sentence, embedding_dict):
    pass


def main():
    word_index_dict = {'UNK': 0, 'SOS': 1, 'EOS': 2, 'PAD': 3, 'apple': 4, 'banana': 5, 'right': 6, 'oov_test': 7}
    save_path = os.path.abspath('../resource/embedding/glove_42b_embed_{}'.format(len(word_index_dict)))
    glove_embedding_path = os.path.abspath('../resource/embedding/glove.42B.300d.txt')
    embed_mat = load_glove_embeddings(glove_embedding_path, word_index_dict, save_path)
    print('util execute accomplished')


if __name__ == '__main__':
    main()
