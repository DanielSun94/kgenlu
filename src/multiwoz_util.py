# The data format source code is from
# https://github.com/alexa/dstqa/blob/master/multiwoz_2.1_format.py
# and https://raw.githubusercontent.com/jasonwu0731/trade-dst/master/utils/fix_label.py
# including fix_general_label_error, utt_format, bs_format
import random
import numpy as np
import os
import pickle
import torch
from multiwoz_config import args, UNK, PAD, CLS, SEP, EXPERIMENT_DOMAINS, UNK_token, PAD_token, CLS_token, \
    SEP_token, DATA_TYPE_SLOT, DATA_TYPE_BELIEF, DATA_TYPE_UTTERANCE, DEVICE, logger
import json
import re
import torch.utils.data as data
from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
# preset token tag
preset_word_num = 4
max_seq_len = args['max_sentence_length']


def prepare_data(read_from_cache, file_path):
    if read_from_cache:
        data_dict, train, dev, test, word_index_stat, slot_value_dict, slot_type_dict, max_utterance_len = \
            pickle.load(open(file_path, 'rb'))
    else:
        batch_size = args['batch_size']
        data_path = args['multiwoz_dataset_folder']
        training_fraction = args['training_data_fraction']
        train_file_path = os.path.join(data_path, 'train_dials.json')
        dev_file_path = os.path.join(data_path, 'dev_dials.json')
        test_file_path = os.path.join(data_path, 'test_dials.json')
        ontology_file_path = os.path.join(data_path, 'ontology.json')
        word_index_stat = None

        all_slots = get_slot_information(ontology_file_path)
        valid_idx_set = dialogue_filter(train_file_path, dev_file_path, test_file_path, args['train_domain'],
                                        args['test_domain'])
        data_dict, max_utterance_len = load_corpus(all_slots, train_file_path, dev_file_path, test_file_path,
                                                   valid_idx_set)
        data_dict, slot_value_dict, slot_type_dict = \
            state_tokenize(data_dict, args['train_domain'], args['test_domain'])
        if args['pretrained_model'] is None:
            word_index_stat = create_word_index_mapping(all_slots, train_file_path, dev_file_path, test_file_path,
                                                        valid_idx_set)
            data_dict = tokenize_utterance(data_dict, word_index_stat)
        else:
            data_dict = tokenize_utterance(data_dict)

        train = get_sequence(data_dict['train'], batch_size, slot_value_dict, slot_type_dict, True, training_fraction)
        dev = get_sequence(data_dict['dev'], batch_size, slot_value_dict, slot_type_dict, True)
        test = get_sequence(data_dict['test'], batch_size, slot_value_dict, slot_type_dict, True)
        pickle.dump([data_dict, train, dev, test, word_index_stat, slot_value_dict, slot_type_dict, max_utterance_len],
                    open(file_path, 'wb'))
    # load_glove_embeddings(word_index_stat.word2index)
    logger.info('data prepared')

    logger.info("Read %s pairs train" % len(data_dict['train']))
    logger.info("Read %s pairs dev" % len(data_dict['dev']))
    logger.info("Read %s pairs test" % len(data_dict['test']))
    # logger.info("Vocab_size: %s " % word_index_stat.n_words)
    logger.info("Max. length of dialog words: %s " % max_utterance_len)
    logger.info("Device = {}".format(DEVICE))
    return train, dev, test, word_index_stat, slot_value_dict, slot_type_dict


def tokenize_utterance(data_dict, word_index_stat=None):

    for dialogues in data_dict['train'], data_dict['dev'], data_dict['test']:
        for dialog in dialogues:
            if args['pretrained_model'] is None:
                context = [word_index_stat.word2index[word] if word_index_stat.word2index.__contains__(word)
                                  else UNK_token for word in dialog['context'].split(' ')]
                current_utterance = [word_index_stat.word2index[word] if word_index_stat.word2index.__contains__(word)
                                     else UNK_token for word in dialog['turn_uttr'].split(' ')]
            elif args['pretrained_model'] == 'roberta':
                context = tokenizer(dialog['context'])['input_ids']
                current_utterance = tokenizer(dialog['turn_uttr'])['input_ids']
            else:
                raise ValueError('invalid pretrained model name')
            dialog['turn_uttr_plain'] = dialog['turn_uttr']
            dialog['context_plain'] = dialog['context']
            dialog['turn_uttr'] = current_utterance
            dialog['context'] = context
    return data_dict


def get_sequence(pairs, batch_size, slot_value_dict, slot_type_dict, sample_type, data_fraction=100):
    data_info, data_info_shuffled = {}, {}
    data_keys = pairs[0].keys()
    shuffle_idx_list = [i for i in range(len(pairs)*data_fraction//100)]
    random.shuffle(shuffle_idx_list)

    for k in data_keys:
        data_info[k] = []
        data_info_shuffled[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    for k in data_keys:
        for idx in shuffle_idx_list:
            data_info_shuffled[k].append(data_info[k][idx])

    dataset = Dataset(data_info_shuffled, slot_value_dict, slot_type_dict)

    if args["multi_gpu"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=16,
                                                 pin_memory=True, drop_last=True, sampler=train_sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                                 shuffle=sample_type, collate_fn=collate_fn)
    return dataloader


def collate_fn(data_):
    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        PAD_Token的index就是1

        if context length is larger than max_len, truncate the context
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        if max_len > max_seq_len:
            max_len = max_seq_len
        assert PAD_token == 1
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            seq = torch.Tensor(seq)
            end = lengths[i]
            if end < max_len:
                padded_seqs[i, :end] = seq[:end]
            else:
                padded_seqs[i] = seq[:max_len]
        padded_seqs = padded_seqs.detach()  # torch.tensor(padded_seqs)
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data_.sort(key=lambda x: len(x['context']), reverse=True)
    item_info = {}
    for key in data_[0].keys():
        item_info[key] = [d[key] for d in data_]

    src_seqs, src_lengths = merge(item_info['context'])
    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths

    # label reorganize
    label_origin = item_info['label']
    slot_type_dict = item_info['slot_type_dict'][0]
    label_reorganize = {}
    for sample in label_origin:
        for slot_name in sample:
            label = None
            if not label_reorganize.__contains__(slot_name):
                label_reorganize[slot_name] = []
            if slot_type_dict[slot_name] == 'classify':
                if sample[slot_name] is not None:
                    label = sample[slot_name]
                else:
                    label =-1
            elif slot_type_dict[slot_name] == 'span':
                # sample[slot_name][1]: end index
                if sample[slot_name] is not None:
                    label = sample[slot_name]
                else:
                    label = -1, -1
            else:
                ValueError('')
            assert label is not None
            label_reorganize[slot_name].append(label)
    for slot_name in label_reorganize:
        label_reorganize[slot_name] = np.array(label_reorganize[slot_name])
    item_info['label'] = label_reorganize

    # gate reorganize
    gate = item_info['gate']
    gate_reorganize = {}
    for index, sample in enumerate(gate):
        for slot_name in sample:
            if not gate_reorganize.__contains__(slot_name):
                gate_reorganize[slot_name] = np.zeros(len(gate), dtype=np.long)
            gate_reorganize[slot_name][index] = int(sample[slot_name])
            assert int(sample[slot_name]) in (0, 1, 2)
    item_info['gate'] = gate_reorganize
    return item_info


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, slot_value_dict, slot_type_dict):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.context = data_info['context']
        self.context_plain = data_info['context_plain']
        self.turn_uttr_plain = data_info['turn_uttr_plain']
        self.turn_belief = data_info['turn_belief']
        self.gate = data_info['gate']
        self.turn_uttr = data_info['turn_uttr']
        self.label = data_info["label"]
        self.slot_value_dict = slot_value_dict
        self.slot_type_dict = slot_type_dict
        self.num_total_seqs = len(self.context)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        id_ = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gate[index]
        turn_uttr = self.turn_uttr[index]
        turn_uttr_plain = self.turn_uttr_plain[index]
        turn_domain = self.preprocess_domain(self.turn_domain[index])
        label = self.label[index]
        context = self.context[index]
        context_plain = self.context_plain[index]

        item_info = {
            "ID": id_,
            "turn_id": turn_id,
            "turn_belief": turn_belief,
            "gate": gating_label,
            "context": context,
            "context_plain": context_plain,
            "turn_uttr": turn_uttr,
            "turn_uttr_plain": turn_uttr_plain,
            "turn_domain": turn_domain,
            'slot_type_dict': self.slot_type_dict,
            'slot_value_dict': self.slot_value_dict,
            "turn_domain_plain": self.turn_domain[index],
            "label": label,
        }
        return item_info

    def __len__(self):
        return self.num_total_seqs

    @staticmethod
    def preprocess(sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story


    @staticmethod
    def preprocess_domain(turn_domain):
        domains = {"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4, "hospital": 5, "bus": 6,
                   "police": 7}
        return domains[turn_domain]


def state_tokenize(dataset: dict, train_domain, test_domain) -> [dict, dict]:
    """
    We apply dual strategy to address DST challenge in this study. That is, if the number of possible values of a slot
    is small than the parameter span_limit (default 10), we will treat the DST problem on the slot as a classification
    problem. Otherwise we will treat the DST problem as a span matching problem.

    Of note, count the fail cases of span match slot (the value does not exist in the utterance)
    :return:
        new_data, a extension data dict with
            gate: 0-3, indicating the type of gate of a turn (not mention, dont care, hit)
            state label: indicating the label of current turn:
                for classify slot, return the index of label
                for span match slot, return the index of start and end word
        classify word index dict
    """
    slot_value_dict, slot_type_dict = dict(), dict()
    span_limit = args['span_limit']
    fail_match = 0
    count = 0

    train_domain = set(train_domain.split('$'))
    test_domain = set(test_domain.split('$'))

    slot_name_idx = 0
    for key in dataset:
        part_of_data = dataset[key]
        for dialogue in part_of_data:
            turn_beliefs = dialogue['turn_belief']
            for turn_belief in turn_beliefs:
                slot_domain = turn_belief.split('-')[0]
                slot_name = turn_belief.split('-')[0] + '-' + turn_belief.split('-')[1]
                slot_value = turn_belief.split('-')[2]
                if key == 'train' or key == 'dev':
                    if slot_domain not in train_domain:
                        continue
                else:
                    if slot_domain not in test_domain:
                        continue
                if not slot_value_dict.__contains__(slot_name):
                    slot_value_dict[slot_name] = set()
                    slot_name_idx += 1
                if slot_value != 'dont care' and slot_value != 'none':
                    slot_value_dict[slot_name].add(slot_value)

    for slot_name in slot_value_dict:
        if len(slot_value_dict[slot_name]) >= span_limit:
            slot_type_dict[slot_name] = 'span'
        else:
            slot_type_dict[slot_name] = 'classify'
        slot_value_dict[slot_name] = sorted(list(slot_value_dict[slot_name]))

    for key in dataset:
        part_of_data = dataset[key]
        for dialogue in part_of_data:
            turn_beliefs = dialogue['turn_belief']
            dialogue['gate'], dialogue['label'] = {}, {}
            for slot_name in slot_type_dict:
                dialogue['gate'][slot_name], dialogue['label'][slot_name] = 0, None
            for turn_belief in turn_beliefs:
                slot_domain = turn_belief.split('-')[0]
                slot_name = turn_belief.split('-')[0] + '-' + turn_belief.split('-')[1]
                slot_value = turn_belief.split('-')[2]
                if key == 'train' or key == 'dev':
                    if slot_domain not in train_domain:
                        continue
                else:
                    if slot_domain not in test_domain:
                        continue
                if slot_value == 'none':
                    dialogue['gate'][slot_name] = 0
                elif slot_value == 'dont care':
                    dialogue['gate'][slot_name] = 1
                else:
                    dialogue['gate'][slot_name] = 2

                if slot_type_dict[slot_name] == 'classify':
                    if not (slot_value == 'none' or slot_value == 'dont care'):
                        dialogue['label'][slot_name] = slot_value_dict[slot_name].index(turn_belief.split('-')[-1])
                else:
                    if not (slot_value == 'none' or slot_value == 'dont care'):
                        count += 1
                        if args['pretrained_model'] is None:
                            value_idx = dialogue['context'].find(' ' + slot_value + ' ')
                            if value_idx == -1:
                                start_idx = -1
                                end_idx = -1
                            else:
                                start_idx = len(dialogue['context'][:value_idx].strip().split(' '))
                                end_idx = start_idx + len(slot_value.strip().split(' ')) - 1
                                # check
                                # pre_context = dialogue['context'][:value_idx]
                                tokenized_context = dialogue['context'].split(' ')
                                for i, item in enumerate(tokenized_context):
                                    if end_idx >= i >= start_idx:
                                        assert item == slot_value.strip().split(' ')[i - start_idx]
                            dialogue['label'][slot_name] = [start_idx, end_idx]
                        elif args['pretrained_model'] == 'roberta':
                            # Roberta uses BPE tokenizer
                            token_list = tokenizer(slot_value)['input_ids'][1: -1]
                            if len(token_list) == 0:
                                logger.info(slot_value)
                            context_tokenize = tokenizer(dialogue['context'])['input_ids']
                            start_idx = -1
                            for i in range(len(context_tokenize)):
                                if  len(token_list) != 0 and context_tokenize[i] == token_list[0] and \
                                        (i+len(token_list)) < len(context_tokenize) \
                                    and context_tokenize[i:i + len(token_list)] == token_list:
                                    start_idx = i
                                    break
                            if start_idx == -1:
                                end_idx = -1
                            else:
                                end_idx = start_idx + len(token_list) -1
                            dialogue['label'][slot_name] = [start_idx, end_idx]

                            # if start_idx == -1:
                            #     logger.info(dialogue['context'])
                            #     logger.info('slot name: {}, slot value: {}'.format(slot_name, slot_value))
                            #     logger.info(context_tokenize)
                            #     logger.info(token_list)
                        else:
                            raise ValueError('invalid pretrained model')
                        if start_idx == -1:
                            fail_match += 1
    logger.info('span fail match count: {}, count: {}, ratio: {}'.format(fail_match, count, fail_match / count))
    return dataset, slot_value_dict, slot_type_dict


def dialogue_filter(train_file_path, dev_file_path, test_file_path, train_domain, test_domain):
    """
    :param test_file_path:
    :param dev_file_path:
    :param train_file_path:
    :param train_domain:
        Indicating the domain of interest in training and dev dataset. Excluding all dialogs whose domain are irrelevant
        to the domain of interest. Of note, as multi-domain dialogue contains several domains, "irrelevant" means every
        domain of a multi-domain dialogue is not in the train_domain, otherwise the dialogue will be reserved
    :param test_domain: Indicating the domain of interest in test dataset
    :return:
    """
    train_domain = set(train_domain.split('$'))
    test_domain = set(test_domain.split('$'))
    valid_dialogue_idx_set = set()
    data_type_dict = {'train': train_file_path, 'dev': dev_file_path, 'test': test_file_path}
    for key in data_type_dict:
        dials = json.load(open(data_type_dict[key], 'r'))
        for dial_dict in dials:
            dialog_id = dial_dict['dialogue_idx']
            domain = set(dial_dict['domains'])
            if key == 'train' or key == 'dev':
                if len(domain.intersection(train_domain)) > 0:
                    valid_dialogue_idx_set.add(dialog_id)
            else:
                if len(domain.intersection(test_domain)) > 0:
                    valid_dialogue_idx_set.add(dialog_id)
    return valid_dialogue_idx_set


def load_corpus(all_slots, train_file, dev_file, test_file, valid_idx_set):
    # max length of slot value string
    max_history_len = 0
    domain_counter = {}
    data_type_dict = {'train': train_file, 'dev': dev_file, 'test': test_file}
    data_dict = {'train': [], 'dev': [], 'test': []}

    for key in data_type_dict:
        dials = json.load(open(data_type_dict[key], 'r'))
        for dial_dict in dials:
            dialog_history = []
            if dial_dict['dialogue_idx'] not in valid_idx_set:
                continue

            for domain in dial_dict["domains"]:
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Reading data
            for ti in range(len(dial_dict["dialogue"])):
                turn = dial_dict['dialogue'][ti]
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                if ti == 0:
                    turn_uttr = utt_format(turn["transcript"] + " " + SEP)
                else:
                    turn_uttr = utt_format(turn["system_transcript"] + " " + SEP + ' ' + turn["transcript"] + " " + SEP)
                turn_uttr_strip = turn_uttr.strip()

                turn_belief_dict = bs_format(fix_general_label_error(turn["belief_state"], False, all_slots))
                turn_belief_list = [str(k) + '-' + str(v) for k, v in turn_belief_dict.items()]

                dialog_history.insert(0, utt_format(turn_uttr))

                source_text = ''
                for i in range(len(dialog_history)):
                    source_text += dialog_history[i].strip() + ' '

                data_detail = {
                    "ID": dial_dict["dialogue_idx"],
                    "domains": dial_dict["domains"],
                    "turn_domain": turn_domain,
                    "turn_id": turn_id,
                    "context": CLS+' ' + source_text,
                    "turn_belief": turn_belief_list,
                    "turn_uttr": CLS+' ' + turn_uttr_strip,
                }
                data_dict[key].append(data_detail)

                if max_history_len < len(source_text.split()):
                    max_history_len = len(source_text.split())
    logger.info('domain count: {}'.format(domain_counter))
    logger.info('corpus loaded')
    return data_dict, max_history_len


def create_word_index_mapping(all_slots, train_file, dev_file, test_file, valid_idx_set):
    """
    check 20210924
    """
    word_index_stat = WordIndexStat()
    word_index_stat.index_words(all_slots, 'slot')

    for file in train_file, dev_file, test_file:
        dialogue_data = json.load(open(file, 'r'))
        for dial_dict in dialogue_data:
            if dial_dict['dialogue_idx'] not in valid_idx_set:
                continue
            for ti, turn in enumerate(dial_dict["dialogue"]):
                word_index_stat.index_words(turn["system_transcript"], 'utterance')
                word_index_stat.index_words(turn["transcript"], 'utterance')
                turn_belief_dict = bs_format(fix_general_label_error(turn["belief_state"], False, all_slots))
                word_index_stat.index_words(turn_belief_dict, 'belief')
    logger.info("word index mapping created")
    return word_index_stat


class WordIndexStat:
    """
    Check 20210921
    build dataset specific word index dict
    """

    def __init__(self):
        self.word2index = {PAD: PAD_token, CLS: CLS_token, SEP: SEP_token, UNK: UNK_token}
        self.index2word = {PAD_token: PAD, CLS_token: CLS, SEP_token: SEP, UNK_token: UNK}
        self.n_words = len(self.index2word)

    def index_words(self, sent, data_type):
        if data_type == DATA_TYPE_UTTERANCE:
            for word in sent.split(" "):
                self.index_word(word)
        elif data_type == DATA_TYPE_SLOT:
            for slot in sent:
                domain, slot = slot.split("-")
                self.index_word(domain)
                for ss in slot.split(" "):
                    self.index_word(ss)
        elif data_type == DATA_TYPE_BELIEF:
            for slot, value in sent.items():
                domain, sole_slot = slot.split("-")
                self.index_word(domain)
                for ss in sole_slot.split(" "):
                    self.index_word(ss)
                for v in value.split(" "):
                    self.index_word(v)
        else:
            raise ValueError('')

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def get_slot_information(ontology_file_path):
    """
    Check 20210924
    enumerate all slots
    """
    ontology = json.load(open(ontology_file_path, 'r'))
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    slots = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return slots


def load_glove_embeddings(word_index_dict):
    """
    The embeddings of out-of-vocab word in the word_index_dict will be assigned as UNK vector
    if the used model has embeddings of UNK, PAD, EOS, SOS. we will use zero vector, average, or random vectors
    to represent them respectively.
    """
    save_path = os.path.join(args['aligned_embedding_path'], 'glove_42B_embed_{}'.format(len(word_index_dict)))
    if os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    for key in {UNK, SEP, CLS, PAD}:
        assert key in word_index_dict

    logger.info("Loading Glove Model")
    glove_embedding = []
    glove_word_index_dict = {}
    with open(args['full_embedding_path'], 'r', encoding='utf-8') as f:
        line_idx = 0
        for line in f:
            split_line = line.split()
            word = split_line[0]
            word_embedding = np.array(split_line[len(split_line) - 300:], dtype=np.float64)
            if len(word_embedding) != 300:
                logger.info(word)
                logger.info(len(split_line))
                logger.info(split_line)
                continue
            glove_word_index_dict[word] = line_idx
            glove_embedding.append(word_embedding)
            line_idx += 1
    logger.info(f"{len(glove_embedding)} words loaded!")
    glove_embedding = np.array(glove_embedding)

    embedding_dimension = len(glove_embedding[0])
    pad_embedding = (np.random.random(embedding_dimension) - 0.5) * 2
    unk_embedding = np.average(glove_embedding, axis=0)
    # I did not find tutorial on how to initialize embedding of sos and eos
    # So I just use zero-mean random vectors
    sep_embedding = (np.random.random(embedding_dimension) - 0.5) * 2
    cls_embedding = (np.random.random(embedding_dimension) - 0.5) * 2

    embedding_mat = np.zeros([len(word_index_dict), embedding_dimension])
    for word in word_index_dict:
        idx = word_index_dict[word]
        if glove_word_index_dict.__contains__(word):
            embedding_mat[idx] = glove_embedding[glove_word_index_dict[word]]
        else:
            if word == PAD:
                embedding_mat[idx] = pad_embedding
            elif word == SEP:
                embedding_mat[idx] = sep_embedding
            elif word == CLS:
                embedding_mat[idx] = cls_embedding
            else:
                embedding_mat[idx] = unk_embedding

    pickle.dump(embedding_mat, open(save_path, 'wb'))
    return embedding_mat


def graph_embeddings_alignment(entity_embed_dict, word_index_dict):
    embedding_dimension = len(entity_embed_dict[list(entity_embed_dict.keys())[0]])
    pad_embedding = np.zeros(embedding_dimension)
    unk_embedding_list = []
    for key in entity_embed_dict:
        unk_embedding_list.append(entity_embed_dict[key])
    unk_embedding = np.average(unk_embedding_list, axis=0)
    # I did not find tutorial on how to initialize embedding of sos and eos
    # So I just use zero-mean random vectors
    cls_embedding = (np.random.random(embedding_dimension) - 0.5) * 2
    sep_embedding = (np.random.random(embedding_dimension) - 0.5) * 2

    embedding_mat = np.zeros([len(word_index_dict), embedding_dimension])
    for word in word_index_dict:
        idx = word_index_dict[word]
        if entity_embed_dict.__contains__(word):
            embedding_mat[idx] = entity_embed_dict[word]
        else:
            if word == PAD:
                embedding_mat[idx] = pad_embedding
            elif word == CLS:
                embedding_mat[idx] = cls_embedding
            elif word == SEP:
                embedding_mat[idx] = sep_embedding
            else:
                embedding_mat[idx] = unk_embedding
    return embedding_mat


def load_graph_embeddings(embed_path, word_net_path):
    """
        load the embeddings of words learned from a semantic knowledge graph (WordNet)
        The embeddings of out-of-vocab word in the word_index_dict will be assigned as UNK vector
        if the used model has embeddings of UNK, PAD, EOS, SOS. we will use zero vector, average, or random vectors
        to represent them respectively.
    """
    entity_embed_dict = {}
    relation_embed_dict = {}
    word_net_obj = pickle.load(open(word_net_path, 'rb'))
    idx_relation_dict = word_net_obj['idx_relation_dict']
    idx_entity_dict = word_net_obj['idx_word_dict']
    entity_embed = torch.load(embed_path)['model_state_dict']['entities_emb.weight'].cpu().numpy()
    relation_embed = torch.load(embed_path)['model_state_dict']['relations_emb.weight'].cpu().numpy()

    for idx in idx_relation_dict:
        relation_embed_dict[idx_relation_dict[idx]] = relation_embed[idx]
    for idx in idx_entity_dict:
        entity_embed_dict[idx_entity_dict[idx]] = entity_embed[idx]
    return entity_embed_dict, relation_embed_dict


def fix_general_label_error(labels, type_, slots):
    label_dict = dict([(l_[0], l_[1]) for l_ in labels]) if type_ else dict(
        [(l_["slots"][0][0], l_["slots"][0][1]) for l_ in labels])

    general_typo = {
        # type
        "guesthouse": "guest house", "guesthouses": "guest house", "guest": "guest house",
        "mutiple sports": "multiple sports",
        "sports": "multiple sports", "mutliple sports": "multiple sports", "swimmingpool": "swimming pool",
        "concerthall": "concert hall",
        "concert": "concert hall", "pool": "swimming pool", "night club": "nightclub", "mus": "museum",
        "ol": "architecture",
        "archaelogy": "archaeology",
        "colleges": "college", "coll": "college", "architectural": "architecture", "musuem": "museum",
        "churches": "church",
        # area
        "center": "centre", "center of town": "centre", "near city center": "centre", "in the north": "north",
        "cen": "centre", "east side": "east",
        "east area": "east", "west part of town": "west", "ce": "centre", "town center": "centre",
        "centre of cambridge": "centre",
        "city center": "centre", "the south": "south", "scentre": "centre", "town centre": "centre",
        "in town": "centre", "north part of town": "north",
        "centre of town": "centre", "cb30aq": "none",
        # price
        "mode": "moderate", "moderate -ly": "moderate", "mo": "moderate",
        # day
        "next friday": "friday", "monda": "monday", "thur": "thursday", "not given": "none",
        # parking
        "free parking": "free",
        # internet
        "free internet": "yes",
        # star
        "4 star": "4", "4 stars": "4", "0 star rarting": "none",
        # others
        "y": "yes", "any": "dontcare", "n": "no", "does not care": "dontcare", "not men": "none", "not": "none",
        "not mentioned": "none",
        '': "none", "not mendtioned": "none", "3 .": "3", "does not": "no", "fun": "none", "art": "none",
        "no mentioned": "none",
        '13.29': '13:29', '1100': '11:00', '11.45': '11:45', '1830': '18:30'
    }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in general_typo.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], general_typo[label_dict[slot]])

            # miss match slot and value
            if slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast",
                                                             "centre", "venetian", "intern", "a cheap -er hotel"] or \
                    slot == "hotel-internet" and label_dict[slot] == "4" or \
                    slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                    slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery",
                                                                       "science", "m"] or \
                    "area" in slot and label_dict[slot] in ["moderate"] or \
                    "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4",
                                                               "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no":
                    label_dict[slot] = "north"
                elif label_dict[slot] == "we":
                    label_dict[slot] = "west"
                elif label_dict[slot] == "cent":
                    label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we":
                    label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no":
                    label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                    slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum",
                                                                       "same area as hotel"]:
                label_dict[slot] = "none"

    return label_dict


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        super(ImbalancedDatasetSampler, self).__init__(dataset)  # test

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    @staticmethod
    def _get_label(dataset, idx):
        return dataset.turn_domain[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def bs_format(bs):
    new_bs = dict()
    for ds, v in bs.items():
        d = ds.split("-")[0]
        s = ds.split("-")[1]

        # drop the first 0 in time
        mat = re.findall(r"(\d{1,2}:\d{1,2})", v)
        if len(mat) == 1 and mat[0][0] == '0':
            v = mat[0][1:]

        if v.__contains__('&'):
            v = v.replace('&', 'and')
        if v.__contains__('archaelogy'):
            v = v.replace('archaelogy', 'archaeology')
        if v == "1730":
            v = "17:30"
        if v == "theater":
            v = "theatre"
        if v == " est ":
            v = " east "
        if v == 'not meavalonntioned':
            v = 'not mentioned'
        if v == "restaurant 2 two":
            v = "2 two"
        if v == "cambridge contemporary art museum":
            v = "cambridge contemporary art"
        if v == "cafe jello museum":
            v = "cafe jello gallery"
        if v == "whippple museum":
            v = "whipple museum"
        if v == "st christs college":
            v = "christ s college"
        if v == "abc theatre":
            v = "adc"
        if d == "train" and v == "london":
            v = "london kings cross"
        if v == "the castle galleries":
            v = "castle galleries"
        if v == "cafe jello":
            v = "cafe jello gallery"
        if v == "cafe uno":
            v = "caffe uno"
        if v == "el shaddia guesthouse":
            v = "el shaddai"
        if v == "kings college":
            v = "king s college"
        if v == "saint johns college":
            v = "saint john s college"
        if v == "kettles yard":
            v = "kettle s yard"
        if v == "grafton hotel":
            v = "grafton hotel restaurant"
        if v == "churchills college":
            v = "churchill college"
        if v == "the churchill college":
            v = "churchill college"
        if v == "portugese":
            v = "portuguese"
        if v == "rosas bed and breakfast":
            v = "rosa s bed and breakfast"
        if v == "pizza hut fenditton":
            v = "pizza hut fen ditton"
        if v == "great saint marys church":
            v = "great saint mary s church"
        if v == "alimentum":
            v = "restaurant alimentum"
        if v == "shiraz":
            v = "shiraz restaurant"
        if v == "christ college":
            v = "christ s college"
        if v == "peoples portraits exhibition at girton college":
            v = "people s portraits exhibition at girton college"
        if v == "saint catharines college":
            v = "saint catharine s college"
        if v == "the maharajah tandoor":
            v = "maharajah tandoori restaurant"
        if v == "efes":
            v = "efes restaurant"
        if v == "the gonvile hotel":
            v = "gonville hotel"
        if v == "abbey pool":
            v = "abbey pool and astroturf pitch"
        if v == "the cambridge arts theatre":
            v = "cambridge arts theatre"
        if v == "sheeps green and lammas land park fen causeway":
            v = "sheep s green and lammas land park fen causeway"
        if v == "rosas bed and breakfast":
            v = "rosa s bed and breakfast"
        if v == "little saint marys church":
            v = "little saint mary s church"
        if v == "pizza hut":
            v = "pizza hut city centre"
        if v == "cambridge contemporary art museum":
            v = "cambridge contemporary art"
        if v == "chiquito":
            v = "chiquito restaurant bar"
        if v == "king hedges learner pool":
            v = "kings hedges learner pool"
        if v == "dontcare":
            v = "dont care"
        if v == "does not care":
            v = "dont care"
        if v == "corsican":
            v = "corsica"
        if v == "barbeque":
            v = "barbecue"
        if v == "center":
            v = "centre"
        if v == "east side":
            v = "east"
        if s == "pricerange":
            s = "price range"
        if s == "price range" and v == "mode":
            v = "moderate"
        if v == "not mentioned":
            v = ""
        if v == "thai and chinese":  # only one such type, throw away
            v = "chinese"
        if s == "area" and v == "n":
            v = "north"
        if s == "price range" and v == "ch":
            v = "cheap"
        if v == "moderate -ly":
            v = "moderate"
        if s == "area" and v == "city center":
            v = "centre"
        # sushi only appear once in the training dataset. doesnt matter throw it away or not
        if s == "food" and v == "sushi":
            v = "japanese"
        if v == "meze bar restaurant":
            v = "meze bar"
        if v == "golden house golden house":
            v = "golden house"
        if v == "fitzbillies":
            v = "fitzbillies restaurant"
        if v == "city stop":
            v = "city stop restaurant"
        if v == "cambridge lodge":
            v = "cambridge lodge restaurant"
        if v == "ian hong house":
            v = "lan hong house"
        if v == "lan hong":
            v = "lan hong house"
        if v == "the americas":
            v = "americas"
        if v == "guest house":
            v = "guesthouse"
        if v == "margherita":
            v = "la margherita"
        if v == "gonville":
            v = "gonville hotel"
        if s == "parking" and v == "free":
            v = "yes"
        if v == "night club":
            v = "nightclub"
        if d == "hotel":
            if v == 'inexpensive':
                v = 'cheap'
        if d == 'restaurant' and s == 'book day':
            if v == 'w':
                v = 'wednesday'
        if d == "hotel" and s == "name":
            if v == "acorn" or v == "acorn house":
                v = "acorn guest house"
            if v == "huntingdon hotel":
                v = "huntingdon marriott hotel"
            if v == "alexander":
                v = "alexander bed and breakfast"
            if v == "university arms":
                v = "university arms hotel"
            if v == "city roomz":
                v = "cityroomz"
            if v == "ashley":
                v = "ashley hotel"
        if d == "train":
            if s == "destination" or s == "departure":
                if v == "bishop stortford":
                    v = "bishops stortford"
                if v == "bishops storford":
                    v = "bishops stortford"
                if v == "birmingham":
                    v = "birmingham new street"
                if v == "stansted":
                    v = "stansted airport"
                if v == "leicaster":
                    v = "leicester"
        if d == "attraction":
            if v == "cambridge temporary art":
                v = "contemporary art museum"
            if v == "cafe jello":
                v = "cafe jello gallery"
            if v == "fitzwilliam" or v == "fitzwilliam museum":
                v = "fitzwilliam"
            if v == "contemporary art museum":
                v = "cambridge contemporary art"
            if v == "christ college":
                v = "christ s college"
            if v == "old school":
                v = "old schools"
            if v == "queen s college":
                v = "queens college"
            if v == "all saint s church":
                v = "all saints church"
            if v == "parkside":
                v = "parkside pools"
            if v == "saint john s college .":
                v = "saint john s college"
            if v == "the mumford theatre":
                v = "mumford theatre"
        if d == "taxi":
            if v == "london kings cross train station":
                v = "london kings cross"
            if v == "stevenage train station":
                v = "stevenage"
            if v == "junction theatre":
                v = "junction"
            if v == "bishops stortford train station":
                v = "bishops stortford"
            if v == "cambridge train station":
                v = "cambridge"
            if v == "citiroomz":
                v = "cityroomz"
            if v == "london liverpool street train station":
                v = "london liverpool street"
            if v == "norwich train station":
                v = "norwich"
            if v == "kings college":
                v = "king s college"
            if v == "the ghandi" or v == "ghandi":
                v = "gandhi"
            if v == "ely train station":
                v = "ely"
            if v == "stevenage train station":
                v = "stevenage"
            if v == "peterborough train station":
                v = "peterborough"
            if v == "london kings cross train station":
                v = "london kings cross"
            if v == "kings lynn train station":
                v = "kings lynn"
            if v == "stansted airport train station":
                v = "stansted airport"
            if v == "acorn house":
                v = "acorn guest house"
            if v == "queen s college":
                v = "queens college"
            if v == "leicester train station":
                v = "leicester"
            if v == "the gallery at 12":
                v = "gallery at 12"
            if v == "caffee uno":
                v = "caffe uno"
            if v == "stevenage train station":
                v = "stevenage"
            if v == "finches":
                v = "finches bed and breakfast"
            if v == "broxbourne train station":
                v = "broxbourne"
            if v == "country folk museum":
                v = "cambridge and county folk museum"
            if v == "ian hong":
                v = "lan hong house"
            if v == "the byard art museum":
                v = "byard art"
            if v == "birmingham new street train station":
                v = "birmingham new street"
            if v == "man on the moon concert hall":
                v = "man on the moon"
            if v == "st . john s college":
                v = "saint john s college"
            if v == "st johns chop house":
                v = "saint johns chop house"
            if v == "maharajah tandoori restaurant4":
                v = "maharajah tandoori restaurant"
            if v == "the soul tree":
                v = "soul tree"
            if v == "aylesbray lodge":
                v = "aylesbray lodge guest house"
            if v == "the alexander bed and breakfast":
                v = "alexander bed and breakfast"
            if v == "shiraz .":
                v = "shiraz restaurant"
            if v == "tranh binh":
                v = "thanh binh"
            if v == "riverboat georginawd":
                v = "riverboat georgina"
            if v == "lovell ldoge":
                v = "lovell lodge"
            if v == "alyesbray lodge hotel":
                v = "aylesbray lodge guest house"
            if v == "wandlebury county park":
                v = "wandlebury country park"
            if v == "the galleria":
                v = "galleria"
            if v == "cambridge artw2orks":
                v = "cambridge artworks"
        new_bs[d + '-' + s] = v
    return new_bs


def utt_format(utt):
    # drop the first 0 in time
    mat = re.findall(r"(\d{1,2}:\d{1,2})", utt)
    if len(mat) > 0:
        for item in mat:
            if item[0] == '0':
                v = item[1:]
                utt = utt.replace(item, v)

    utt = utt.replace("theater", "theatre")
    utt = utt.replace("barbeque", "barbecue")
    utt = utt.replace("center", "centre")
    utt = utt.replace("swimmingpool", "swimming pool")
    utt = utt.replace("italain", "italian")
    utt = utt.replace("da vinci pizzria", "da vinci pizzeria")
    utt = utt.replace("concerthall", "concert hall")
    utt = utt.replace("artchway house", "archway house")
    utt = utt.replace("carribean", "caribbean")
    utt = utt.replace("traveller s rest", "travellers rest")
    utt = utt.replace("thrusday", "thursday")
    utt = utt.replace("night club", "nightclub")
    utt = utt.replace("michael house cafe", "michaelhouse cafe")
    utt = utt.replace("riverside brasseri", "riverside brasserie")
    utt = utt.replace("riverside brassiere", "riverside brasserie")
    utt = utt.replace("cambidge", "cambridge")
    utt = utt.replace("cambrdige", "cambridge")
    utt = utt.replace("king s", "kings")
    utt = utt.replace("queen s", "queens")
    utt = utt.replace("cambride towninfo centre", "cambridge towninfo centre")
    utt = utt.replace("stantsted airport", "stansted airport")
    utt = utt.replace("jingling noodle bar", "jinling noodle bar")
    utt = utt.replace("longon", "london")
    utt = utt.replace("guesthouse", "guest house")
    utt = utt.replace("cincema", "cinema")
    utt = utt.replace("&", "and")
    utt = utt.replace("lecester", "leicester")
    utt = utt.replace('nando s', 'nandos')
    utt = utt.replace('siagon city', 'saigon city')
    return utt


def main():
    file_path = args['cache_path']
    train, dev, test, word_index_stat, slot_value_dict, slot_type_dict = prepare_data(False, file_path)

    wordnet_embedding_path = args['wordnet_embedding_path']
    wordnet_path = args['wordnet_path']

    entity_embed_dict, relation_embed_dict = load_graph_embeddings(wordnet_embedding_path, wordnet_path)
    graph_embeddings_alignment(entity_embed_dict, word_index_stat.word2index)
    logger.info('util execute accomplished')


if __name__ == '__main__':
    main()