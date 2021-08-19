import json
import torch
import torch.utils.data as data
import random
from multiwoz_config import PAD_token, SOS_token, EOS_token, UNK_token, USE_CUDA, args, multiwoz_data_folder,\
    multiwoz_resource_folder, EXPERIMENT_DOMAINS
from collections import OrderedDict
from embeddings import GloveEmbedding, KazumaCharEmbedding
from tqdm import tqdm
import os
import pickle
from random import shuffle
from multiwoz_fix_label import fix_general_label_error


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent, type_):
        if type_ == 'utter':
            for word in sent.split(" "):
                self.index_word(word)
        elif type_ == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
        elif type_ == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
                for v in value.split(" "):
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.gating_label = data_info['gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.generate_y = data_info["generate_y"]
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        id_ = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        turn_uttr = self.turn_uttr[index]
        # 此处做了Token->Index转换
        turn_domain = self.preprocess_domain(self.turn_domain[index])
        generate_y = self.generate_y[index]
        # 此处做了Token->Index转换
        generate_y = self.preprocess_slot(generate_y, self.trg_word2id)
        context = self.dialog_history[index]
        # 此处做了Token->Index转换
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]

        item_info = {
            "ID": id_,
            "turn_id": turn_id,
            "turn_belief": turn_belief,
            "gating_label": gating_label,
            "context": context,
            "context_plain": context_plain,
            "turn_uttr_plain": turn_uttr,
            "turn_domain": turn_domain,
            "generate_y": generate_y,
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
    def preprocess_slot(sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    @staticmethod
    def preprocess_memory(sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book", "").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story

    @staticmethod
    def preprocess_domain(turn_domain):
        domains = {"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4, "hospital": 5, "bus": 6,
                   "police": 7}
        return domains[turn_domain]


def collate_fn(data_):
    def merge(sequences):
        """
        从不定长的Token list整合为定长的Token Input (with padding)
        merge from batch * sent_len to batch * max_len
        PAD_Token的index就是1
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach()  # torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        """
        从不定长的Token list整合为定长的Token Input (with padding)
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        """
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l_) for l_ in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len - len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    # def merge_memory(sequences):
    #     lengths = [len(seq) for seq in sequences]
    #     max_len = 1 if max(lengths) == 0 else max(lengths)  # avoid the empty belief state issue
    #     padded_seqs = torch.ones(len(sequences), max_len, 4).long()
    #     for i, seq in enumerate(sequences):
    #         end = lengths[i]
    #         if len(seq) != 0:
    #             padded_seqs[i, :end, :] = seq[:end]
    #     return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data_.sort(key=lambda x: len(x['context']), reverse=True)
    item_info = {}
    for key in data_[0].keys():
        item_info[key] = [d[key] for d in data_]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
    gating_label = torch.tensor(item_info["gating_label"])
    turn_domain = torch.tensor(item_info["turn_domain"])

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        gating_label = gating_label.cuda()
        turn_domain = turn_domain.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths
    return item_info


def read_lang(file_name, gating_dict, all_slots, dataset, lang, mem_lang, training, max_line=None):
    """
    所有的utterance都携带了完整的上下文信息
    """
    print(("Reading from {}".format(file_name)))
    data_ = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = {}
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first
        for dial_dict in dials:
            if dataset == "train" and training:
                # 对语料中的词进行编号
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')
        # determine training data ratio, default is 100%
        if training and dataset == "train" and args["data_ratio"] != 100:
            random.Random(10).shuffle(dials)
            dials = dials[:int(len(dials) * 0.01 * args["data_ratio"])]

        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""
            # last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            # only domain是仅保留某个主题的dialog，如果是多主题对话，则至少要包含目标主题，否则跳过
            # except的逻辑比较复杂
            #   在Training和Validation Dataset中，其含义是，对单领域的对话，如果是except的，则删除
            #   在Test中，其意义是，没他则删除
            #   为了zero shot learning这是很显然的数据准备策略，训练数据没他，测试数据有他
            if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in
                dial_dict["domains"]) or (args["except_domain"] != "" and dataset != "test" and
                                          [args["except_domain"]] == dial_dict["domains"]):
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                turn_uttr_strip = turn_uttr.strip()
                dialog_history += (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
                source_text = dialog_history.strip()
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, all_slots)

                # Generate domain-dependent slot list
                slot_temp = all_slots
                # 根据前面那句话的逻辑，如果一个dialog是多domain的，其中有个domain是except domain中的，这样的dialog不会被跳过去
                # 但是这也的确造成了一些问题，一个是训练时标签泄露，另一个是测试时测试不需要的domain，因此需要进行修正
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "":
                        slot_temp = [k for k in all_slots if args["except_domain"] not in k]
                        turn_belief_dict = OrderedDict(
                            [(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                    # 原代码中是elif，改为if
                    # elif args["only_domain"] != "":
                    if args["only_domain"] != "":
                        slot_temp = [k for k in all_slots if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict(
                            [(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "":
                        slot_temp = [k for k in all_slots if args["except_domain"] in k]
                        turn_belief_dict = OrderedDict(
                            [(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                    if args["only_domain"] != "":
                        slot_temp = [k for k in all_slots if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict(
                            [(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                turn_belief_list = [str(k) + '-' + str(v) for k, v in turn_belief_dict.items()]

                if dataset == "train" and training:
                    mem_lang.index_words(turn_belief_dict, 'belief')

                class_label, generate_y, slot_mask, gating_label = [], [], [], []
                # start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys():
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])

                data_detail = {
                    "ID": dial_dict["dialogue_idx"],
                    "domains": dial_dict["domains"],
                    "turn_domain": turn_domain,
                    "turn_id": turn_id,
                    # 每一个中间态的dial均包含完整的历史数据，以方便训练时组织数据
                    "dialog_history": source_text,
                    "turn_belief": turn_belief_list,
                    "gating_label": gating_label,
                    "turn_uttr": turn_uttr_strip,
                    # 本轮提到的slot的具体值
                    'generate_y': generate_y
                }
                data_.append(data_detail)

                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())

            # 这个max line过界退出的功能是原始代码就有的，我也不知道有什么用，就留着不动吧，反正也没什么影响
            cnt_lin += 1
            if max_line and cnt_lin >= max_line:
                break

    # add t{} to the lang file
    if "t{}".format(max_value_len - 1) not in mem_lang.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter')

    print("domain_counter", domain_counter)
    return data_, max_resp_len, slot_temp


def get_seq(pairs, lang, mem_lang, batch_size, type_):
    if type_ and args['fisher_sample'] > 0:
        shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    dataset = Dataset(data_info, lang.word2index, lang.word2index, mem_lang.word2index)

    if args["imbalance_sampler"] and type_:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  # shuffle=type,
                                                  collate_fn=collate_fn,
                                                  sampler=ImbalancedDatasetSampler(dataset))
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=type_,
                                                  collate_fn=collate_fn)
    return data_loader


def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    es = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        es.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(es, f)


def get_slot_information(ontology):
    """枚举出所有slot"""
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    slots = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return slots


def prepare_data_seq(training, batch_size):
    """
    取消了training set和dev, validation set之间batch size的设置差异，使用统一的batch size
    """
    file_train = os.path.join(multiwoz_data_folder, 'train_dials.json')
    file_dev = os.path.join(multiwoz_data_folder, 'dev_dials.json')
    file_test = os.path.join(multiwoz_data_folder, 'test_dials.json')

    if args['add_name'] != '':
        save_folder = os.path.join(multiwoz_resource_folder, 'save/{}-'.format(args["decoder"] + args["add_name"]))
    else:
        save_folder = os.path.join(multiwoz_resource_folder, 'save/{}'.format(args["decoder"]))
    print("folder_name", save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # load domain-slot pairs from ontology
    # 此处的ontology其实指代了所有可能出现的slot value
    ontology = json.load(open(os.path.join(multiwoz_data_folder, 'ontology.json'), 'r'))
    all_slots = get_slot_information(ontology)
    # 指代slot可能出现的三种情况，ptr代指有说明，dontcare代表用户无所谓，none代表未提及
    gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
    # Vocabulary, 分别给slot进行idx word匹配
    # lang专管utterance的index word，mem专管slot和相对应的值得index word
    lang, mem_lang = Lang(), Lang()
    lang.index_words(all_slots, 'slot')
    mem_lang.index_words(all_slots, 'slot')
    # lang-all.pkl, mem-lang-all.pkl 存储的主要是index word dict
    lang_name = 'lang-all.pkl'
    mem_lang_name = 'mem-lang-all.pkl'

    if training:
        pair_train, train_max_len, slot_train = read_lang(file_train, gating_dict, all_slots, "train", lang, mem_lang,
                                                          training)
        train = get_seq(pair_train, lang, mem_lang, batch_size, True)
        nb_train_vocab = lang.n_words
        pair_dev, dev_max_len, slot_dev = read_lang(file_dev, gating_dict, all_slots, "dev", lang, mem_lang, training)
        dev = get_seq(pair_dev, lang, mem_lang, batch_size, False)
        pair_test, test_max_len, slot_test = read_lang(file_test, gating_dict, all_slots, "test", lang, mem_lang,
                                                       training)
        test = get_seq(pair_test, lang, mem_lang, batch_size, False)

        if os.path.exists(save_folder + lang_name) and os.path.exists(save_folder + mem_lang_name):
            print("[Info] Loading saved lang files...")
            with open(save_folder + lang_name, 'rb') as handle:
                lang = pickle.load(handle)
            with open(save_folder + mem_lang_name, 'rb') as handle:
                mem_lang = pickle.load(handle)
        else:
            print("[Info] Dumping lang files...")
            with open(save_folder + lang_name, 'wb') as handle:
                pickle.dump(lang, handle)
            with open(save_folder + mem_lang_name, 'wb') as handle:
                pickle.dump(mem_lang, handle)
        emb_dump_path = os.path.join(multiwoz_data_folder, 'emb{}.json'.format(len(lang.index2word)))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            # embed dump load 只做一次
            # 根据 index2word 的表去查embedding，保证dump的json中的list index对应的词embedding和index word dict中的index
            # embedding 是一致的
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)
    else:
        with open(save_folder + lang_name, 'rb') as handle:
            lang = pickle.load(handle)
        with open(save_folder + mem_lang_name, 'rb') as handle:
            mem_lang = pickle.load(handle)

        pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
        pair_dev, dev_max_len, slot_dev = read_lang(file_dev, gating_dict, all_slots, "dev", lang, mem_lang, training)
        dev = get_seq(pair_dev, lang, mem_lang, batch_size, False)
        pair_test, test_max_len, slot_test = read_lang(file_test, gating_dict, all_slots, "test", lang, mem_lang,
                                                       training)
        test = get_seq(pair_test, lang, mem_lang, batch_size, False)

    test_4d = []
    if args['except_domain'] != "":
        pair_test_4d, _, _ = read_lang(file_test, gating_dict, all_slots, "dev", lang, mem_lang, training)
        test_4d = get_seq(pair_test_4d, lang, mem_lang, batch_size, False)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % nb_train_vocab)
    print("Vocab_size Belief %s" % mem_lang.n_words)
    print("Max. length of dialog words for RNN: %s " % max_word)
    print("USE_CUDA={}".format(USE_CUDA))

    slots_list = [all_slots, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(slots_list[2]))))
    print(slots_list[2])
    print("[Test Set Slots]: Number is {} in total".format(str(len(slots_list[3]))))
    print(slots_list[3])
    lang_list = [lang, mem_lang]
    return train, dev, test, test_4d, lang_list, slots_list, gating_dict, nb_train_vocab


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


def unit_test():
    train, dev, test, test_special, lang, slots_list, gating_dict, max_word = \
        prepare_data_seq(True, args['batch_size'])
    p_bar = tqdm(enumerate(train), total=len(train))
    for i, batch_data in p_bar:
        if i % 200 == 0:
            print(i)


if __name__ == '__main__':
    unit_test()
