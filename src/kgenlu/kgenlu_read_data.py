import os
import pickle
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
from kgenlu_config import args, logger, dev_idx_path, test_idx_path, act_data_path, ACT_SLOT_NAME_MAP_DICT, \
    label_normalize_path, dialogue_data_cache_path, dialogue_data_path, SEP_token, CLS_token, DOMAIN_IDX_DICT, \
    SLOT_IDX_DICT, ACT_MAP_DICT, UNK_token, dialogue_unstructured_data_cache_path, PAD_token, \
    classify_slot_value_index_map_path, DEVICE
from transformers import RobertaTokenizer

# state matching的部分是否能够都对上，什么_booking之类的
# 其实这样子做标签会非常稀疏，但是因为Trippy也用了类似的做法，那我们也用类似的做法
overwrite_cache = False
if args['pretrained_model'] == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
else:
    raise ValueError('')
NORMALIZE_MAP = json.load(open(label_normalize_path, 'r'))
domain_slot_list = NORMALIZE_MAP['slots']
domain_index_map = NORMALIZE_MAP['domain_index']
slot_index_map = NORMALIZE_MAP['slot_index']
domain_slot_type_map = NORMALIZE_MAP['slots-type']
label_normalize_map = NORMALIZE_MAP['label_maps']
train_domain = args['train_domain']
test_domain = args['test_domain']
use_history = args['use_history']
no_value_assign_strategy = args['no_value_assign_strategy']
max_len = args['max_len']
train_domain_set = set(train_domain.strip().split('$'))
test_domain_set = set(test_domain.strip().split('$'))
active_slot_count = dict()
unpointable_slot_value_set = set()


def prepare_data(overwrite):
    train_idx_list, dev_idx_list, test_idx_list = get_dataset_idx(dev_idx_path, test_idx_path, dialogue_data_path)
    idx_dict = {'train': train_idx_list, 'dev': dev_idx_list, 'test': test_idx_list}

    dialogue_data = json.load(open(dialogue_data_path, 'r'))
    act_data = json.load(open(act_data_path, 'r'))
    if os.path.exists(classify_slot_value_index_map_path) and not overwrite:
        classify_slot_value_index_dict, classify_slot_index_value_dict = \
            pickle.load(open(classify_slot_value_index_map_path, 'rb'))
    else:
        classify_slot_value_index_dict, classify_slot_index_value_dict = \
            get_classify_slot_index_map(idx_dict, dialogue_data, act_data)
    data = [process_data(idx_dict['train'], dialogue_data, act_data, 'train', classify_slot_value_index_dict,
                         overwrite),
            process_data(idx_dict['dev'], dialogue_data, act_data, 'dev', classify_slot_value_index_dict,
                         overwrite),
            process_data(idx_dict['test'], dialogue_data, act_data, 'test', classify_slot_value_index_dict,
                         overwrite)]
    return data, classify_slot_value_index_dict, classify_slot_index_value_dict


def get_classify_slot_index_map(idx_dict, dialogue_dict, act_data):
    idx_set = set(idx_dict['train'] + idx_dict['dev'] + idx_dict['test'])
    raw_data_dict = {}
    for dialogue_idx in dialogue_dict:
        if dialogue_idx not in idx_set:
            continue
        if dialogue_idx.strip().split('.')[0] not in act_data:
            logger.info('act of {} not found'.format(dialogue_idx))
        utterance_list, state_dict, act_dict = get_dialogue_info(act_data, dialogue_dict, dialogue_idx)
        raw_data_dict[dialogue_idx] = utterance_list, state_dict, act_dict
    classify_slot_value_index_dict, classify_slot_index_value_dict = classify_slot_value_indexing(raw_data_dict)
    return classify_slot_value_index_dict, classify_slot_index_value_dict


def process_data(idx_list, dialogue_dict, act, data_type, classify_slot_value_index_dict, overwrite):
    if os.path.exists(dialogue_unstructured_data_cache_path.format(data_type)) and not overwrite:
        data_dict = pickle.load(open(dialogue_unstructured_data_cache_path.format(data_type), 'rb'))
    else:
        data_dict = {}
        raw_data_dict = {}
        idx_set = set(idx_list)
        for dialogue_idx in dialogue_dict:
            if dialogue_idx not in idx_set:
                continue
            if dialogue_idx.strip().split('.')[0] not in act:
                logger.info('act of {} not found'.format(dialogue_idx))

            utterance_list, state_dict, act_dict = get_dialogue_info(act, dialogue_dict, dialogue_idx)
            raw_data_dict[dialogue_idx] = utterance_list, state_dict, act_dict

        for dialogue_idx in raw_data_dict:
            utterance_list, state_dict, act_dict = raw_data_dict[dialogue_idx]
            data_dict[dialogue_idx] = \
                dialogue_reorganize_and_normalize(utterance_list, state_dict, act_dict, classify_slot_value_index_dict)
        pickle.dump(data_dict, open(dialogue_unstructured_data_cache_path.format(data_type), 'wb'))

    logger.info(active_slot_count)
    logger.info('data reorganized, starting transforming data to the model required format')
    if os.path.exists(dialogue_data_cache_path.format(data_type)) and not overwrite:
        processed_data = pickle.load(open(dialogue_data_cache_path.format(data_type), 'rb'))
    else:
        if data_type == 'train' or 'dev':
            processed_data = prepare_data_for_model(data_dict, max_len, classify_slot_value_index_dict,
                                                    train_domain_set)
        else:
            assert data_type == 'test'
            processed_data = prepare_data_for_model(data_dict, max_len, classify_slot_value_index_dict, test_domain_set)
        logger.info('prepare process finished')
        pickle.dump(processed_data, open(dialogue_data_cache_path.format(data_type), 'wb'))
    logger.info('constructing dataloader')
    dataloader = construct_dataloader(processed_data)
    return dataloader


def collate_fn(batch):
    sample_id_list, active_domain_list, active_slot_list, turn_change_label_list, turn_cumulative_label_list, \
        context_token_id_list, referred_list_dict, hit_type_list_dict, hit_value_list_dict, context_mask_list = \
        [], [], [], [], [], [], {}, {}, {}, []
    for domain_slot in domain_slot_list:
        referred_list_dict[domain_slot] = []
        hit_type_list_dict[domain_slot] = []
        hit_value_list_dict[domain_slot] = []
    for sample in batch:
        sample_id_list.append(sample[0])
        active_domain_list.append(sample[1])
        active_slot_list.append(sample[2])
        turn_change_label_list.append(sample[3])
        turn_cumulative_label_list.append(sample[4])
        context_token_id_list.append(sample[5])
        context_mask_list.append(sample[6])
        for domain_slot in domain_slot_list:
            hit_type_list_dict[domain_slot].append(sample[7][domain_slot])
            hit_value_list_dict[domain_slot].append(sample[8][domain_slot])
            referred_list_dict[domain_slot].append(sample[9][domain_slot])

    active_domain_list = torch.FloatTensor(active_domain_list)
    active_slot_list = torch.FloatTensor(active_slot_list)
    context_token_id_list = torch.LongTensor(context_token_id_list)
    context_mask_list = torch.BoolTensor(context_mask_list)
    for domain_slot in domain_slot_list:
        hit_type_list_dict[domain_slot] = torch.LongTensor(hit_type_list_dict[domain_slot])
        hit_value_list_dict[domain_slot] = torch.LongTensor(hit_value_list_dict[domain_slot])
        referred_list_dict[domain_slot] = torch.LongTensor(referred_list_dict[domain_slot])
    return sample_id_list, active_domain_list, active_slot_list, turn_change_label_list, turn_cumulative_label_list,\
        context_token_id_list, referred_list_dict, hit_type_list_dict, hit_value_list_dict, context_mask_list


def construct_dataloader(processed_data):
    sample_id_list = [item.sample_id for item in processed_data]
    active_domain_list = [item.active_domain for item in processed_data]
    active_slot_list = [item.active_slot for item in processed_data]
    turn_change_label_list = [item.turn_change_label for item in processed_data]
    turn_cumulative_label_list = [item.turn_cumulative_labels for item in processed_data]
    context_list = [item.context for item in processed_data]
    context_mask_list = [item.context_mask for item in processed_data]
    hit_type_list_dict, hit_value_list_dict, referred_list_dict = {}, {}, {}
    for domain_slot in domain_slot_list:
        hit_type_list_dict[domain_slot] = [item.hit_type[domain_slot] for item in processed_data]
        hit_value_list_dict[domain_slot] = [item.hit_value[domain_slot] for item in processed_data]
        referred_list_dict[domain_slot] = [item.referred[domain_slot] for item in processed_data]
    dataset = SampleDataset(sample_id_list, active_domain_list, active_slot_list, turn_change_label_list,
                            turn_cumulative_label_list, context_list, context_mask_list, hit_type_list_dict,
                            hit_value_list_dict, referred_list_dict)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args['batch_size'], collate_fn=collate_fn)
    return dataloader


class SampleDataset(Dataset):
    def __init__(self, sample_id_list, active_domain_list, active_slot_list, turn_change_label_list,
                 turn_cumulative_label_list, context_list, context_mask_list, hit_type_list_dict, hit_value_list_dict,
                 referred_list_dict):
        self.sample_id_list = sample_id_list
        self.active_domain_list = active_domain_list
        self.active_slot_list = active_slot_list
        self.turn_change_label_list = turn_change_label_list
        self.turn_cumulative_label_list = turn_cumulative_label_list
        self.context_list = context_list
        self.context_mask_list = context_mask_list
        self.hit_type_list_dict = hit_type_list_dict
        self.hit_value_list_dict = hit_value_list_dict
        self.referred_list_dict = referred_list_dict

    def __getitem__(self, index):
        sample_id = self.sample_id_list[index]
        active_domain = self.active_domain_list[index]
        active_slot = self.active_slot_list[index]
        turn_cumulative_label = self.turn_cumulative_label_list[index]
        turn_change_label = self.turn_change_label_list[index]
        context = self.context_list[index]
        context_mask = self.context_mask_list[index]
        hit_type_dict, hit_value_dict, referred_dict = {}, {}, {}
        for domain_slot in domain_slot_list:
            hit_type_dict[domain_slot] = self.hit_type_list_dict[domain_slot][index]
            hit_value_dict[domain_slot] = self.hit_value_list_dict[domain_slot][index]
            referred_dict[domain_slot] = self.referred_list_dict[domain_slot][index]
        return sample_id, active_domain, active_slot, turn_change_label, turn_cumulative_label, context, context_mask, \
            hit_type_dict, hit_value_dict, referred_dict

    def __len__(self):
        return len(self.sample_id_list)


class Sample(object):
    def __init__(self, sample_id, active_domain, active_slot, filtered_state, context, context_mask,
                 turn_change_label, turn_cumulative_labels):
        self.sample_id = sample_id
        self.active_domain = active_domain
        self.active_slot = active_slot
        self.turn_change_label = turn_change_label
        self.turn_cumulative_labels = turn_cumulative_labels
        self.context = context
        self.context_mask = context_mask
        self.hit_type, self.hit_value, self.referred = {}, {}, {}
        for domain_slot in domain_slot_list:
            self.hit_type[domain_slot] = filtered_state[domain_slot]['hit_type']
            self.referred[domain_slot] = filtered_state[domain_slot]['referred_slot_idx']
            if domain_slot_type_map[domain_slot] == 'classify':
                assert isinstance(filtered_state[domain_slot]['hit_value'], int)
                self.hit_value[domain_slot] = filtered_state[domain_slot]['hit_value']
            else:
                hit_value = filtered_state[domain_slot]['hit_value']
                assert isinstance(hit_value, tuple) and len(hit_value) == 2
                self.hit_value[domain_slot] = hit_value[0], hit_value[1]


def prepare_data_for_model(data_dict, max_input_length, classify_slot_value_index_dict, interest_domain):
    data_for_model = []
    for dialogue_idx in data_dict:
        dialogue = data_dict[dialogue_idx]
        for turn_idx in dialogue:
            turn_data = dialogue[turn_idx]
            turn_active_domains = active_domain_structurize(turn_data['active_domain'])
            turn_active_slots = active_slot_structurize(turn_data['active_slots'])
            turn_utterance_token, turn_utterance_token_map_list = \
                utterance_tokenize(turn_data['current_turn_utterance_token'])
            turn_history_utterance_token, turn_history_utterance_token_map_list \
                = utterance_tokenize(turn_data['history_utterance_token'])

            context, context_label_dict, context_mask = joint_alignment_and_truncate(
                turn_history_utterance_token, turn_history_utterance_token_map_list, turn_utterance_token,
                turn_utterance_token_map_list, turn_data['state'], max_input_length)
            turn_state = state_structurize(turn_data['state'], context_label_dict, classify_slot_value_index_dict)
            filtered_state = irrelevant_domain_label_mask(turn_state, interest_domain)
            turn_change_label = turn_data['turn_change_state']
            turn_cumulative_labels = turn_data['turn_cumulative_labels']

            data_for_model.append(
                Sample(sample_id=dialogue_idx+'-'+str(turn_idx),
                       active_domain=turn_active_domains,
                       active_slot=turn_active_slots,
                       filtered_state=filtered_state,
                       turn_change_label=turn_change_label,
                       turn_cumulative_labels=turn_cumulative_labels,
                       context=context,
                       context_mask=context_mask)
            )
    return data_for_model


def irrelevant_domain_label_mask(turn_state, interest_domain):
    filtered_turn_state = {}
    for domain_slot in domain_slot_list:
        filtered_turn_state[domain_slot] = {}
        if domain_slot.strip().split('-')[0] not in interest_domain:
            filtered_turn_state[domain_slot]['hit_type'] = -1
            filtered_turn_state[domain_slot]['referred_slot_idx'] = -1
            if domain_slot_type_map[domain_slot] == 'classify':
                filtered_turn_state[domain_slot]['hit_value'] = -1
            else:
                filtered_turn_state[domain_slot]['hit_value'] = -1, -1
        else:
            filtered_turn_state[domain_slot]['hit_type'] = turn_state[domain_slot]['hit_type']
            filtered_turn_state[domain_slot]['referred_slot_idx'] = turn_state[domain_slot]['referred_slot_idx']
            filtered_turn_state[domain_slot]['hit_value'] = turn_state[domain_slot]['hit_value']
    return filtered_turn_state


def joint_alignment_and_truncate(turn_history_utterance_token, turn_history_utterance_token_map_list,
                                 turn_utterance_token, turn_utterance_token_map_list, turn_state, max_input_length):
    context = turn_utterance_token + [0] + turn_history_utterance_token
    context_label_dict = {}
    for domain_slot in domain_slot_list:
        token_label = turn_state[domain_slot]['token_label']
        history_token_label = turn_state[domain_slot]['history_token_label']
        aligned_turn_label, aligned_history_label = [], []
        assert turn_utterance_token_map_list[-1] == len(token_label) - 1
        assert turn_history_utterance_token_map_list[-1] == len(history_token_label) - 1

        for origin_index in turn_utterance_token_map_list:
            aligned_turn_label.append(token_label[origin_index])
        for origin_index in turn_history_utterance_token_map_list:
            aligned_history_label.append(history_token_label[origin_index])
        if not use_history:
            aligned_history_label = [0 for _ in aligned_history_label]
        context_label_dict[domain_slot] = aligned_turn_label + [0] + aligned_history_label

    if len(context) > max_input_length:
        context = context[: max_input_length]
        for domain_slot in domain_slot_list:
            context_label_dict[domain_slot] = context_label_dict[domain_slot][: max_input_length]
        context_mask = [1 for _ in range(max_input_length)]
    else:
        padding_num = max_input_length - len(context)
        padding_token = tokenizer.convert_tokens_to_ids([' '+PAD_token])
        context_mask = [0] * len(context) + [1] * padding_num
        context = context + padding_token * padding_num
        for domain_slot in domain_slot_list:
            context_label_dict[domain_slot] = context_label_dict[domain_slot] + [0] * padding_num
        assert len(context_mask) == max_input_length

    for domain_slot in domain_slot_list:
        assert len(context) == len(context_label_dict[domain_slot])
    return context, context_label_dict, context_mask


def active_domain_structurize(active_domains):
    active_domain_list = [0 for _ in range(len(domain_index_map))]
    for domain in active_domains:
        active_domain_list[domain_index_map[domain]] = 1
    return active_domain_list


def active_slot_structurize(active_slots):
    active_slot_list = [0 for _ in range(len(slot_index_map))]
    for slot in active_slots:
        active_slot_list[slot_index_map[slot]] = 1
    return active_slot_list


def utterance_tokenize(utterance_token):
    origin_token_idx, new_token_list, new_token_origin_token_map_list = 0, list(), list()
    for token in utterance_token:
        new_tokens = tokenizer.tokenize(' ' + token)
        assert len(new_tokens) >= 1
        for new_token in new_tokens:
            new_token_list.append(new_token)
            new_token_origin_token_map_list.append(origin_token_idx)
        origin_token_idx += 1
    new_token_list = tokenizer.convert_tokens_to_ids(new_token_list)
    return new_token_list, new_token_origin_token_map_list


def state_structurize(state, context_label_dict, classify_slot_value_index_dict):
    reorganized_state = {}
    for domain_slot in state:
        if domain_slot not in reorganized_state:
            reorganized_state[domain_slot] = {}
        class_type = state[domain_slot]['class_type']
        referred_slot = state[domain_slot]['referred_slot']
        classify_value_index = state[domain_slot]['classify_value']

        # initialize
        if no_value_assign_strategy == 'miss':
            referred_slot_idx = -1
            if domain_slot_type_map[domain_slot] == 'classify':
                hit_value = -1
            else:
                hit_value = -1, -1
        elif no_value_assign_strategy == 'value':
            referred_slot_idx = 30
            if domain_slot_type_map[domain_slot] == 'classify':
                hit_value = len(classify_slot_value_index_dict[domain_slot])
            else:
                hit_value = 0, 0
        else:
            raise ValueError('')

        # class_type_label
        if class_type == 'none' or class_type == 'unpointable':
            hit_type = 0
        elif class_type == 'dontcare':
            hit_type = 1
        elif class_type == 'inform':
            hit_type = 2
        elif class_type == 'referral':
            hit_type = 3
            referred_slot_idx = domain_slot_list.index(referred_slot)
        else:
            assert class_type == 'hit'
            hit_type = 4
            if domain_slot_type_map[domain_slot] == 'classify':
                hit_value = classify_value_index

        # 注意，因为history的补充label，因此span case的标签标记是和class type无关，终归是要做的
        if domain_slot_type_map[domain_slot] == 'span':
            hit_value = span_idx_extract(context_label_dict, domain_slot, no_value_assign_strategy)

        reorganized_state[domain_slot]['hit_value'] = hit_value
        reorganized_state[domain_slot]['referred_slot_idx'] = referred_slot_idx
        reorganized_state[domain_slot]['hit_type'] = hit_type
    return reorganized_state


def span_idx_extract(context_label_dict, domain_slot, no_value_assign):
    if no_value_assign == 'miss':
        start_idx, end_idx = -1, -1
    else:
        assert no_value_assign == 'value'
        start_idx, end_idx = 0, 0
    if domain_slot_type_map[domain_slot] == 'span':
        if 1 in context_label_dict[domain_slot]:
            start_idx = context_label_dict[domain_slot].index(1)
            if 0 not in context_label_dict[domain_slot][start_idx:]:
                end_idx = len(context_label_dict[domain_slot][start_idx:]) + start_idx - 1
            else:
                end_idx = context_label_dict[domain_slot][start_idx:].index(0) + start_idx - 1
    else:
        assert domain_slot_type_map[domain_slot] == 'classify'
        assert np.sum(np.array(context_label_dict[domain_slot])) == 0
    return start_idx, end_idx


def get_span_index(turn_utterance_token_map_list, origin_utterance_label):
    assert len(origin_utterance_label) == turn_utterance_token_map_list[-1] + 1
    assert np.sum(np.array(origin_utterance_label)) > 0
    new_label = [0 for _ in turn_utterance_token_map_list]
    for index in range(len(turn_utterance_token_map_list)):
        origin_index = turn_utterance_token_map_list[index]
        if origin_utterance_label[origin_index] == 1:
            new_label[index] = 1

    start_index, end_index = -1, -1
    for index in range(len(new_label)):
        if new_label[index] == 1:
            if start_index < 0:
                start_index = index
            if index > end_index:
                end_index = index
    assert start_index != -1 and end_index != -1
    return start_index, end_index


def classify_slot_value_indexing(data_dict):
    classify_slot_value_index_dict, classify_slot_index_value_dict = {}, {}
    for domain_slot in domain_slot_list:
        classify_slot_value_index_dict[domain_slot] = {}
        classify_slot_index_value_dict[domain_slot] = {}
    for dialogue_idx in data_dict:
        for turn in data_dict[dialogue_idx][1]:
            state_dict, _ = state_extract(None, data_dict[dialogue_idx][1][turn])
            for domain_slot in state_dict:
                if domain_slot_type_map[domain_slot] != 'classify':
                    continue
                classify_value = state_dict[domain_slot]
                if classify_value == 'none' or classify_value in classify_slot_value_index_dict[domain_slot]:
                    continue
                idx = len(classify_slot_value_index_dict[domain_slot])
                classify_slot_value_index_dict[domain_slot][classify_value] = idx
                classify_slot_index_value_dict[domain_slot][idx] = classify_value
    return classify_slot_value_index_dict, classify_slot_index_value_dict


def normalize_data(utterance_list, state_dict, act_dict):
    for i in range(len(utterance_list)):
        utterance_list[i] = normalize_text(utterance_list[i])

    return utterance_list, state_dict, act_dict


def dialogue_reorganize_and_normalize(utterance_list, state_dict, act_dict, classify_slot_value_index_dict):
    reorganize_data = {}
    assert len(utterance_list) % 2 == 0
    cumulative_labels = {slot: 'none' for slot in domain_slot_list}

    history = ''
    history_token = []
    history_token_label = {}
    for domain_slot in domain_slot_list:
        history_token_label[domain_slot] = []
    for turn_idx in range(0, len(utterance_list)//2):
        # 注意，数据中的state是积累值，而在设想中，label标签的是本轮的改变状态
        # Trippy的设定中，一方面他用了label repeat的设定，但是另一方面又设计了极为复杂的处理策略把label repeat的影响消除。
        # 我们干脆就不要这样做了，直接只用本轮对话和上轮的总结信息，判定信息的更新，而不是积累的信息

        # We will change the user-system turn to system-user turn for the requirement of NLU task
        # Therefore we will add a empty system utterance at first and discard the last system utterance (it
        # typically doesn't contain useful information)
        # And the act of first turn is empty and the act of last turn will be discard
        reorganize_data[turn_idx] = {}
        value_dict, inform_dict, class_type_dict, referral_dict = {}, {}, {}, {}

        active_domain, active_slots, inform_info, inform_slot_filled_dict = \
            act_reorganize_and_normalize(act_dict, turn_idx, args['auxiliary_domain_assign'])
        if turn_idx == 0:
            assert 1 not in active_domain and 1 not in active_slots

        reorganize_data[turn_idx]['active_domain'] = active_domain
        reorganize_data[turn_idx]['active_slots'] = active_slots

        last_turn_label = cumulative_labels.copy()
        modified_slots, cumulative_labels = state_extract(cumulative_labels, state_dict[turn_idx+1])

        system_utterance = normalize_text('' if turn_idx == 0 else utterance_list[2*turn_idx-1].lower())
        system_utterance = delex_text(system_utterance, inform_info)
        user_utterance = normalize_text(utterance_list[2*turn_idx].lower())
        system_utterance_token, user_utterance_token = tokenize(system_utterance), tokenize(user_utterance)
        current_turn_utterance = user_utterance + ' ' + SEP_token + ' ' + system_utterance
        current_turn_utterance_token = user_utterance_token + [SEP_token] + system_utterance_token

        reorganize_data[turn_idx]['current_turn_utterance'] = CLS_token + ' ' + current_turn_utterance + ' ' + SEP_token
        reorganize_data[turn_idx]['current_turn_utterance_token'] =\
            [CLS_token] + current_turn_utterance_token + [SEP_token]

        reorganize_data[turn_idx]['history_utterance'] = CLS_token + ' ' + history
        reorganize_data[turn_idx]['history_utterance_token'] = [CLS_token] + history_token
        reorganize_data[turn_idx]['state'] = {}

        for domain_slot in domain_slot_list:
            reorganize_data[turn_idx]['state'][domain_slot] = {}

            value_label = 'none'
            if domain_slot in modified_slots:
                value_label = modified_slots[domain_slot]

            inform_label = inform_info[domain_slot] if domain_slot in inform_info else 'none'

            _, referred_slot, user_utterance_token_label, class_type, classify_value = \
                get_turn_label(value_label, inform_label, user_utterance_token, domain_slot, last_turn_label,
                               classify_slot_value_index_dict)

            # Generally don't use span prediction on sys utterance (but inform prediction instead).
            system_utterance_token_label = [0 for _ in system_utterance_token]

            # for cls token and sep token
            turn_utterance_token_label = user_utterance_token_label + [0] + system_utterance_token_label

            assert len(turn_utterance_token_label) == \
                   len(system_utterance_token) + len(user_utterance_token) + 1

            if class_type == 'unpointable':
                class_type_dict[domain_slot] = 'none'
                referral_dict[domain_slot] = 'none'
            else:
                class_type_dict[domain_slot] = class_type
                referral_dict[domain_slot] = referred_slot

            reorganize_data[turn_idx]['state'][domain_slot]['class_type'] = class_type
            reorganize_data[turn_idx]['state'][domain_slot]['classify_value'] = classify_value
            reorganize_data[turn_idx]['state'][domain_slot]['token_label'] = [0] + turn_utterance_token_label + [0]
            reorganize_data[turn_idx]['state'][domain_slot]['referred_slot'] = referred_slot
            reorganize_data[turn_idx]['state'][domain_slot]['history_token_label'] = \
                [0] + history_token_label[domain_slot]
            reorganize_data[turn_idx]['turn_change_state'] = modified_slots
            reorganize_data[turn_idx]['turn_cumulative_labels'] = cumulative_labels

            assert len(history_token_label[domain_slot]) == len(history_token)

            history_token_label[domain_slot] = turn_utterance_token_label + [0] + \
                history_token_label[domain_slot]

        history = current_turn_utterance + ' ' + SEP_token + ' ' + history
        history_token = current_turn_utterance_token + [SEP_token] + history_token
    return reorganize_data


def get_turn_label(value_label, inform_label, user_utterance_token, domain_slot, seen_slots,
                   classify_slot_value_index_dict):
    # four types of class info has it's priority
    # 尽可能提供补充标签
    user_utterance_token_label = [0 for _ in user_utterance_token]
    informed_value = 'none'
    referred_slot = 'none'
    value_index = -1
    if value_label == 'none' or value_label == 'dontcare':
        class_type = value_label
    else:
        in_user_utterance_flag, position, value_index = \
            check_label(value_label, user_utterance_token, domain_slot, classify_slot_value_index_dict)
        is_informed, informed_value = check_slot_inform(value_label, inform_label)
        referred_slot = check_slot_referral(value_label, domain_slot, seen_slots)

        if in_user_utterance_flag:
            start_idx, end_idx = position[-1]
            # if the slot is referred multi times, only use the latest item
            for i in range(start_idx, end_idx):
                user_utterance_token_label[i] = 1

        if is_informed:
            class_type = 'inform'
        elif referred_slot != 'none':
            class_type = 'referral'
        else:
            if domain_slot_type_map[domain_slot] == 'span':
                if in_user_utterance_flag:
                    class_type = 'hit'
                else:
                    class_type = 'unpointable'
            else:
                assert domain_slot_type_map[domain_slot] == 'classify'
                class_type = 'hit'
    return informed_value, referred_slot, user_utterance_token_label, class_type, value_index


def check_slot_referral(value_label, slot, seen_slots):
    referred_slot = 'none'
    if slot == 'hotel-stars' or slot == 'hotel-internet' or slot == 'hotel-parking':
        return referred_slot
    for s in seen_slots:
        # Avoid matches for slots that share values with different meaning.
        # hotel-internet and -parking are handled separately as Boolean slots.
        if s == 'hotel-stars' or s == 'hotel-internet' or s == 'hotel-parking':  # 这些是classification问题
            continue
        if re.match("(hotel|restaurant)-book-people", s) and slot == 'hotel-book-stay':
            continue
        if re.match("(hotel|restaurant)-book-people", slot) and s == 'hotel-book-stay':
            continue
        if slot != s and (slot not in seen_slots or seen_slots[slot] != value_label):
            if seen_slots[s] == value_label:
                referred_slot = s
                break
            elif value_label in label_normalize_map:
                for value_label_variant in label_normalize_map[value_label]:
                    if seen_slots[s] == value_label_variant:
                        referred_slot = s
                        break
    return referred_slot


def check_label(value_label, user_utterance_token, domain_slot, classify_slot_value_index_dict):
    in_user_utterance_flag, position, value_index = False, [], -1

    if domain_slot_type_map[domain_slot] == 'classify':
        value_index = classify_slot_value_index_dict[domain_slot][value_label]

    else:
        assert domain_slot_type_map[domain_slot] == 'span'
        in_user_utterance_flag, position = get_token_position(user_utterance_token, value_label)
        # If no hit even though there should be one, check for value label variants
        if not in_user_utterance_flag and value_label in label_normalize_map:
            for value_label_variant in label_normalize_map[value_label]:
                # 使用了极为复杂的语义判定策略，从而进行判定
                in_user_utterance_flag, position = get_token_position(user_utterance_token, value_label_variant)
                if in_user_utterance_flag:
                    break
        if not in_user_utterance_flag:
            unpointable_slot_value_set.add(value_label)
    return in_user_utterance_flag, position, value_index


def get_token_position(token_list, value_label):
    position = []  # the token may be matched multi times
    found = False
    label_list = [item for item in map(str.strip, re.split("(\W+)", value_label)) if len(item) > 0]
    len_label = len(label_list)
    for i in range(len(token_list) + 1 - len_label):
        if token_list[i:i + len_label] == label_list:
            position.append((i, i + len_label))  # start, exclusive_end
            found = True
    return found, position


def check_slot_inform(value_label, inform_label):
    # 在做inform时极大程度的放宽了match的判定，只要能够碰上都算，而不是要求严格一致
    result = False
    informed_value = 'none'
    value_label = ' '.join(tokenize(value_label))
    if value_label == inform_label:
        result = True
    elif is_in_list(inform_label, value_label):
        result = True
    elif is_in_list(value_label, inform_label):
        result = True
    elif inform_label in label_normalize_map:
        for inform_label_variant in label_normalize_map[inform_label]:
            if value_label == inform_label_variant:
                result = True
                break
            elif is_in_list(inform_label_variant, value_label):
                result = True
                break
            elif is_in_list(value_label, inform_label_variant):
                result = True
                break
    elif value_label in label_normalize_map:
        for value_label_variant in label_normalize_map[value_label]:
            if value_label_variant == inform_label:
                result = True
                break
            elif is_in_list(inform_label, value_label_variant):
                result = True
                break
            elif is_in_list(value_label_variant, inform_label):
                result = True
                break
    if result:
        informed_value = inform_label
    return result, informed_value


def is_in_list(token, value):
    found = False
    token_list = [item for item in map(str.strip, re.split("(\W+)", token)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
    token_len = len(token_list)
    value_len = len(value_list)
    for i in range(token_len + 1 - value_len):  # if the value len is larger than token len, the loop will be skipped
        if token_list[i:i + value_len] == value_list:
            found = True
            break
    return found


def delex_text(utterance, values, unk_token=UNK_token):
    utt_norm = tokenize(utterance)
    for slot, value in values.items():
        if value != 'none':
            v_norm = tokenize(value)
            v_len = len(v_norm)
            for i in range(len(utt_norm) + 1 - v_len):
                if utt_norm[i:i + v_len] == v_norm:
                    utt_norm[i:i + v_len] = [unk_token] * v_len
    utterance = ''
    for item in utt_norm:
        utterance += item + ' '
    return utterance.strip()


def tokenize(utterance):
    utt_lower = normalize_text(utterance)
    utt_tok = [tok for tok in map(str.strip, re.split("(\W+)", utt_lower)) if len(tok) > 0]
    return utt_tok


def state_extract(cumulative_labels, state_dict):
    modified_slots = {}
    for domain in state_dict:
        booked = state_dict[domain]['book']['booked']
        booked_slots = {}
        # Check the booked section
        if booked:
            for slot in booked[0]:
                booked_slots[slot] = normalize_label('{}-{}'.format(domain, slot), booked[0][slot])  # normalize labels
        # Check the semi and the inform slots
        for category in ['book', 'semi']:
            for slot in state_dict[domain][category]:  # s for slot name
                s = '{}-book-{}'.format(domain, slot) if category == 'book' else '{}-{}'.format(domain, slot)
                value_label = normalize_label(s, state_dict[domain][category][slot])
                # Prefer the slot value as stored in the booked section
                if slot in booked_slots:
                    value_label = booked_slots[slot]
                # Remember modified slots and entire dialog state
                if cumulative_labels is not None:
                    if s in domain_slot_list and cumulative_labels[s] != value_label:
                        modified_slots[s] = value_label
                        cumulative_labels[s] = value_label
                else:
                    if s in domain_slot_list:
                        modified_slots[s] = value_label
    if cumulative_labels is not None:
        return modified_slots.copy(), cumulative_labels.copy()
    else:
        return modified_slots.copy(), None


def act_reorganize_and_normalize(act_dict, turn_idx, auxiliary_domain_assign):
    """
    由于数据被左移了一位，第一轮也就是turn_idx = 0 时，system utterance是空的，对应的utterance是空的。
    在整理后的数据所有的act turn全部向右移位一次
    而最后一个turn的action是greeting，被删除。因此，尽管原先dialogue act的turn从1起数，而我们编号中从零起数，但是act没有必要移位
    """
    active_domain, active_slots, inform_info, inform_slot_filled_dict = set(), set(), dict(), dict()
    for domain_slot in domain_slot_list:
        inform_slot_filled_dict[domain_slot] = 0
    if turn_idx not in act_dict or not isinstance(act_dict[turn_idx], dict):
        #  the act dict is string "no annotation" in some cases
        return active_domain, active_slots, inform_info, inform_slot_filled_dict

    turn_act_dict = act_dict[turn_idx]
    for act_name in turn_act_dict:
        # assign active domain and slots
        act_domain, act_type = act_name.lower().strip().split('-')
        domains = [act_domain]
        # if the domain is booking, assign the domain to other valid value
        if domains[0] == 'booking' and auxiliary_domain_assign:
            domains = assign_domain_when_booking(act_dict, turn_idx)
            if len(domains) == 0:
                continue

        for domain in domains:
            if domain in DOMAIN_IDX_DICT:
                active_domain.add(domain)

        act_info = turn_act_dict[act_name]
        for item in act_info:
            slot = item[0].lower().strip()
            normalized_name = None
            if slot in ACT_SLOT_NAME_MAP_DICT:
                normalized_name = ACT_SLOT_NAME_MAP_DICT[slot]
            elif slot in SLOT_IDX_DICT:
                normalized_name = slot
            if normalized_name is not None:
                if normalized_name not in active_slot_count:
                    active_slot_count[normalized_name] = 1
                else:
                    active_slot_count[normalized_name] += 1
                active_slots.add(normalized_name)

    # parse act
    for act_name in turn_act_dict:
        act_domain, act_type = act_name.lower().strip().split('-')
        act_info = turn_act_dict[act_name]
        # assign act label
        if act_type == 'inform' or act_type == 'recommend' or act_type == 'select' or act_type == 'book':
            if act_domain == 'booking':  # did not consider booking case
                continue
            for item in act_info:
                slot = item[0].lower().strip()
                value = item[1].lower().strip()
                if slot == 'none' or value == '?' or value == 'none':
                    continue
                domain_slot = act_domain + '-' + slot
                if domain_slot in ACT_MAP_DICT:
                    domain_slot = ACT_MAP_DICT[domain_slot]

                # In case of multiple mentioned values...
                # ... Option 1: Keep first informed value
                if domain_slot not in inform_info:
                    inform_info[domain_slot] = value

    # overwrite act if it has booking value
    for act_name in turn_act_dict:
        act_domain, act_type = act_name.lower().strip().split('-')
        act_info = turn_act_dict[act_name]
        # assign act label
        if act_type == 'inform' or act_type == 'recommend' or act_type == 'select' or act_type == 'book':
            if act_domain != 'booking':
                continue
            for item in act_info:
                slot = item[0].lower().strip()
                value = item[1].lower().strip()
                if slot == 'none' or value == '?' or value == 'none':
                    continue
                domain_slot = act_domain + '-' + slot
                if domain_slot in ACT_MAP_DICT:
                    domain_slot = ACT_MAP_DICT[domain_slot]
                domain_slots = []
                slot = domain_slot.strip().split('-')[1]
                for name in NORMALIZE_MAP['slots']:
                    if len(active_domain) > 0:
                        if name.__contains__(slot):
                            domain_slots.append(name)
                    else:
                        if name.__contains__(slot):
                            domain_slots.append(name)

                # If the booking slot is already filled skip
                for domain_slot in domain_slots:
                    if domain_slot not in inform_info:
                        inform_info[domain_slot] = normalize_label(domain_slot, value)

    for domain_slot in domain_slot_list:
        if domain_slot in inform_info:
            inform_slot_filled_dict[domain_slot] = 1
    return active_domain, active_slots, inform_info, inform_slot_filled_dict


def assign_domain_when_booking(act_dict, turn_idx):
    # strategy: if current turn has other domain whose name is not booking, assign the target domain to the name
    # otherwise assign the previous turn domain, and then the next turn
    return_domain, origin_turn_idx = [], turn_idx
    while True:
        current_turn_dict = act_dict[turn_idx]
        if isinstance(current_turn_dict, dict):
            for name in current_turn_dict:
                domain, act_type = name.lower().strip().split('-')
                if domain != 'booking' and domain in {'hotel', 'taxi', 'restaurant', 'attraction', 'train'}:
                    return_domain.append(domain)

            if len(return_domain) > 0:
                return return_domain

        if turn_idx <= origin_turn_idx:
            turn_idx -= 1
        else:
            turn_idx += 1

        if turn_idx == 0:
            if origin_turn_idx == len(act_dict):
                break
            else:
                turn_idx = origin_turn_idx + 1
        elif turn_idx > len(act_dict):
            break
    return return_domain


def get_dialogue_info(act, dialogue_dict, dialogue_idx):
    utterance_list, state_dict, act_dict = [], {}, {}
    switch_flag = True
    turn_idx = 0
    for idx, turn in enumerate(dialogue_dict[dialogue_idx]['log']):
        is_system_utterance = turn['metadata'] != {}
        if switch_flag == is_system_utterance:
            logger.info("Wrong order of utterances. Skipping rest of dialog {}".format(dialogue_idx))
            break
        switch_flag = is_system_utterance

        if is_system_utterance:
            turn_idx += 1
            dialogue_state = turn['metadata']
            if str(turn_idx) not in act[dialogue_idx.strip().split('.')[0]]:
                turn_act = {}
            else:
                turn_act = act[dialogue_idx.strip().split('.')[0]][str(turn_idx)]
            state_dict[turn_idx] = dialogue_state
            act_dict[turn_idx] = turn_act

        utterance_list.append(turn['text'])
    return utterance_list, state_dict, act_dict


def get_dataset_idx(dev_path, test_path, dialogue_data):
    # get the dialogue name index, e.g., PMUL3233.json, of train, dev, and test dataset.
    dev_idx_list, test_idx_list = [], []
    with open(dev_path, 'r') as file:
        for line in file:
            dev_idx_list.append(line[:-1])
    with open(test_path, 'r') as file:
        for line in file:
            test_idx_list.append(line[:-1])
    dev_idx_set, test_idx_set, train_idx_set = set(dev_idx_list), set(test_idx_list), set()
    data = json.load(open(dialogue_data, 'r'))
    for key in data:
        if key not in dev_idx_set and key not in test_idx_set:
            train_idx_set.add(key)
    dev_idx_list, test_idx_list, train_idx_list = \
        sorted(list(dev_idx_set)), sorted(list(test_idx_set)), sorted(list(train_idx_set))
    logger.info('train dialogue: {}, dev dialogue: {}, test dialogue: {}'
                .format(len(train_idx_list), len(dev_idx_list), len(test_idx_list)))
    return train_idx_list, dev_idx_list, test_idx_list


def get_slot_list(all_slot_list, train_set, test_set):
    train_slot_set, test_slot_set, all_slot_set = set(), set(), set()
    for slot in all_slot_list:
        for key in train_set:
            if slot.__contains__(key):
                train_slot_set.add(slot)
        for key in test_set:
            if slot.__contains__(key):
                test_slot_set.add(key)

    train_slot_list = sorted(list(train_slot_set))
    test_slot_list = sorted(list(test_slot_set))
    logger.info('train slot size: {}, list: {}'.format(len(train_slot_list), train_slot_list))
    logger.info('test slot size: {}, list: {}'.format(len(test_slot_list), test_slot_list))
    return train_slot_list, test_slot_list


def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text)  # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text)  # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5",
                  text)  # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text)  # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4",
                  text)  # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text)  # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?",
                  lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) +
                  x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text)  # Correct times that use 24 as hour
    return text


def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text)  # Systematic typo
    text = re.sub("guesthouse", "guest house", text)  # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text)  # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text)  # Normalization
    return text


# This should only contain label normalizations. All other mappings should
# be defined in LABEL_MAPS.
def normalize_label(slot, value_label):
    # 根据设计，不同的slot对应不同的标准化方法（比如时间和其他的就不一样），因此要输入具体的slot name
    # Normalization of empty slots
    if value_label == '' or value_label == "not mentioned":
        return "none"

    # Normalization of time slots
    if "leaveat" in slot or "arriveby" in slot or slot == 'restaurant-book_time':
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub("guesthouse", "guest house", value_label)

    return value_label


def main():
    logger.info('label_map load success')
    data, classify_slot_value_index_dict, classify_slot_index_value_dict = prepare_data(overwrite_cache)
    train_loader, dev_loader, test_loader = data
    batch_count = 0
    for _ in tqdm(train_loader):
        batch_count += 1
    for _ in tqdm(dev_loader):
        batch_count += 1
    for _ in tqdm(test_loader):
        batch_count += 1
    print(batch_count)
    logger.info('data read success')

    print(unpointable_slot_value_set)


if __name__ == '__main__':
    main()
