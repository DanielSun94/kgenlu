import os
import pickle
import json
import torch
import math
import re
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import random
from base_config import args, logger, dev_idx_path, test_idx_path, act_data_path, ACT_SLOT_NAME_MAP_DICT, \
    label_normalize_path, dialogue_data_cache_path, dialogue_data_path, SEP_token, CLS_token, DOMAIN_IDX_DICT, \
    SLOT_IDX_DICT, ACT_MAP_DICT, UNK_token, PAD_token, classify_slot_value_index_map_path, UNNORMALIZED_ACTION_SLOT
from transformers import RobertaTokenizer

# state matching的部分是否能够都对上，什么_booking之类的
# 其实这样子做标签会非常稀疏，但是因为Trippy也用了类似的做法，那我们也用类似的做法
if 'roberta' in args['pretrained_model']:
    tokenizer = RobertaTokenizer.from_pretrained(args['pretrained_model'])
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
use_multiple_gpu = args['multi_gpu']
data_fraction = args['train_data_fraction']
no_value_assign_strategy = args['no_value_assign_strategy']
max_len = args['max_len']
delex_system_utterance = args['delex_system_utterance']
variant_flag = args['use_variant']
train_domain_set = set(train_domain.strip().split('$'))
test_domain_set = set(test_domain.strip().split('$'))
active_slot_count = dict()
unpointable_slot_value_list = []


def prepare_data(overwrite):
    logger.info('start prepare data, overwrite flag: {}'.format(overwrite))
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
        pickle.dump([classify_slot_value_index_dict, classify_slot_index_value_dict],
                    open(classify_slot_value_index_map_path, 'wb'))
    dev_data = process_data(idx_dict['dev'], dialogue_data, act_data, 'dev', classify_slot_value_index_dict, overwrite)
    test_data = process_data(idx_dict['test'], dialogue_data, act_data, 'test', classify_slot_value_index_dict,
                             overwrite)
    train_data = process_data(idx_dict['train'], dialogue_data, act_data, 'train', classify_slot_value_index_dict,
                              overwrite)
    data = train_data, dev_data, test_data
    logger.info('data prepared')
    return data, classify_slot_value_index_dict, classify_slot_index_value_dict


def get_classify_slot_index_map(idx_dict, dialogue_dict, act_data):
    # checked
    idx_set = set(idx_dict['train'] + idx_dict['dev'] + idx_dict['test'])
    raw_data_dict = {}
    for dialogue_idx in dialogue_dict:
        if dialogue_idx not in idx_set:
            raise ValueError('')
        if dialogue_idx.strip().split('.')[0] not in act_data:
            logger.info('act of {} not found'.format(dialogue_idx))
        raw_data_dict[dialogue_idx] = get_dialogue_info(act_data, dialogue_dict, dialogue_idx)
    classify_slot_value_index_dict, classify_slot_index_value_dict = classify_slot_value_indexing(raw_data_dict)
    return classify_slot_value_index_dict, classify_slot_index_value_dict


def process_data(idx_list, dialogue_dict, act, data_type, classify_slot_value_index_dict, overwrite):
    if os.path.exists(dialogue_data_cache_path.format(data_type)) and not overwrite:
        dataset = pickle.load(open(dialogue_data_cache_path.format(data_type), 'rb'))
    else:
        data_dict = {}
        raw_data_dict = {}
        idx_set = set(idx_list)
        for dialogue_idx in dialogue_dict:
            if dialogue_idx not in idx_set:
                continue
            if dialogue_idx.strip().split('.')[0] not in act:
                logger.info('act of {} not found'.format(dialogue_idx))
            raw_data_dict[dialogue_idx] = get_dialogue_info(act, dialogue_dict, dialogue_idx)

        for dialogue_idx in raw_data_dict:
            utterance_list, state_dict, act_dict = raw_data_dict[dialogue_idx]
            data_dict[dialogue_idx] = dialogue_reorganize_and_normalize(dialogue_idx, utterance_list, state_dict,
                                                                        act_dict, classify_slot_value_index_dict)

        logger.info(active_slot_count)
        logger.info('data reorganized, starting transforming data to the model required format')
        state_hit_count(data_dict, data_type)
        if data_type == 'train' or 'dev':
            processed_data = prepare_data_for_model(data_dict, max_len, classify_slot_value_index_dict,
                                                    train_domain_set, data_type)
        else:
            assert data_type == 'test'
            processed_data = prepare_data_for_model(data_dict, max_len, classify_slot_value_index_dict, test_domain_set,
                                                    data_type)
        logger.info('prepare process finished')
        logger.info('constructing dataloader')
        dataset = construct_dataloader(processed_data, data_type)
        pickle.dump(dataset, open(dialogue_data_cache_path.format(data_type), 'wb'))

    if data_type == 'train':
        assert 0.01 <= float(data_fraction) <= 1
        dataset = SampleDataset(*dataset.get_fraction_data(float(data_fraction)))

    if use_multiple_gpu and data_type == 'train':
        sampler = DistributedSampler(dataset)
    else:
        if data_type == 'train':
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args['batch_size'], collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    sample_id_list, active_domain_list, active_slot_list, turn_change_label_list, turn_cumulative_label_list, \
        context_token_id_list, referred_list_dict, hit_type_list_dict, hit_value_list_dict, context_mask_list, \
        last_turn_state_list, inform_value_list = [], [], [], [], [], [], {}, {}, {}, [], [], []
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
        last_turn_state_list.append(sample[7])
        inform_value_list.append(sample[11])
        for domain_slot in domain_slot_list:
            hit_type_list_dict[domain_slot].append(sample[8][domain_slot])
            hit_value_list_dict[domain_slot].append(sample[9][domain_slot])
            referred_list_dict[domain_slot].append(sample[10][domain_slot])

    active_domain_list = torch.FloatTensor(active_domain_list)
    active_slot_list = torch.FloatTensor(active_slot_list)
    context_token_id_list = torch.LongTensor(context_token_id_list)
    context_mask_list = torch.BoolTensor(context_mask_list)
    for domain_slot in domain_slot_list:
        hit_type_list_dict[domain_slot] = torch.LongTensor(hit_type_list_dict[domain_slot])
        hit_value_list_dict[domain_slot] = torch.LongTensor(hit_value_list_dict[domain_slot])
        referred_list_dict[domain_slot] = torch.LongTensor(referred_list_dict[domain_slot])
    return sample_id_list, active_domain_list, active_slot_list, turn_change_label_list, turn_cumulative_label_list,\
        context_token_id_list, referred_list_dict, hit_type_list_dict, hit_value_list_dict, context_mask_list, \
        last_turn_state_list, inform_value_list


def construct_dataloader(processed_data, data_type):
    # check
    sample_id_list = [item.sample_id for item in processed_data]
    active_domain_list = [item.active_domain for item in processed_data]
    active_slot_list = [item.active_slot for item in processed_data]
    turn_change_label_list = [item.turn_change_label for item in processed_data]
    turn_cumulative_label_list = [item.turn_cumulative_labels for item in processed_data]
    context_list = [item.context for item in processed_data]
    context_mask_list = [item.context_mask for item in processed_data]
    last_turn_label = [item.last_turn_label for item in processed_data]
    inform_value_list = [item.inform_value for item in processed_data]
    hit_type_list_dict, hit_value_list_dict, referred_list_dict = {}, {}, {}
    for domain_slot in domain_slot_list:
        hit_type_list_dict[domain_slot] = [item.hit_type[domain_slot] for item in processed_data]
        hit_value_list_dict[domain_slot] = [item.hit_value[domain_slot] for item in processed_data]
        referred_list_dict[domain_slot] = [item.referred[domain_slot] for item in processed_data]

    # 此处由于train可以默认为知道上一轮结果真值，因此可以shuffle。而dev和test不知道，需要依赖预测进行判断，因此必须
    if data_type == 'train':
        idx_list = [i for i in range(len(sample_id_list))]
        random.shuffle(idx_list)
        new_sample_id_list, new_active_domain_list, new_active_slot_list, new_turn_change_label_list, \
            new_turn_cumulative_label_list, new_context_list, new_context_mask_list, new_last_turn_label,\
            new_hit_type_list_dict, new_hit_value_list_dict, new_referred_list_dict, new_inform_value_list = \
            [], [], [], [], [], [], [], [], {}, {}, {}, []
        for domain_slot in domain_slot_list:
            new_hit_type_list_dict[domain_slot] = []
            new_hit_value_list_dict[domain_slot] = []
            new_referred_list_dict[domain_slot] = []
        for idx in idx_list:
            new_sample_id_list.append(sample_id_list[idx])
            new_active_domain_list.append(active_domain_list[idx])
            new_active_slot_list.append(active_slot_list[idx])
            new_turn_change_label_list.append(turn_change_label_list[idx])
            new_turn_cumulative_label_list.append(turn_cumulative_label_list[idx])
            new_context_list.append(context_list[idx])
            new_last_turn_label.append(last_turn_label[idx])
            new_context_mask_list.append(context_mask_list[idx])
            new_inform_value_list.append(inform_value_list[idx])
            for domain_slot in domain_slot_list:
                new_hit_type_list_dict[domain_slot].append(hit_type_list_dict[domain_slot][idx])
                new_hit_value_list_dict[domain_slot].append(hit_value_list_dict[domain_slot][idx])
                new_referred_list_dict[domain_slot].append(referred_list_dict[domain_slot][idx])
        dataset = \
            SampleDataset(new_sample_id_list, new_active_domain_list, new_active_slot_list, new_turn_change_label_list,
                          new_turn_cumulative_label_list, new_context_list, new_context_mask_list, new_last_turn_label,
                          new_hit_type_list_dict, new_hit_value_list_dict, new_referred_list_dict,
                          new_inform_value_list)
    else:
        dataset = SampleDataset(sample_id_list, active_domain_list, active_slot_list, turn_change_label_list,
                                turn_cumulative_label_list, context_list, context_mask_list, last_turn_label,
                                hit_type_list_dict, hit_value_list_dict, referred_list_dict, inform_value_list)
    return dataset


class SampleDataset(Dataset):
    # check
    def __init__(self, sample_id_list, active_domain_list, active_slot_list, turn_change_label_list,
                 turn_cumulative_label_list, context_list, context_mask_list, last_turn_label, hit_type_list_dict,
                 hit_value_list_dict, referred_list_dict, inform_value_list):
        self.sample_id_list = sample_id_list
        self.active_domain_list = active_domain_list
        self.active_slot_list = active_slot_list
        self.turn_change_label_list = turn_change_label_list
        self.turn_cumulative_label_list = turn_cumulative_label_list
        self.context_list = context_list
        self.context_mask_list = context_mask_list
        self.last_turn_label = last_turn_label
        self.hit_type_list_dict = hit_type_list_dict
        self.hit_value_list_dict = hit_value_list_dict
        self.referred_list_dict = referred_list_dict
        self.inform_value_list = inform_value_list

    def __getitem__(self, index):
        sample_id = self.sample_id_list[index]
        active_domain = self.active_domain_list[index]
        active_slot = self.active_slot_list[index]
        turn_cumulative_label = self.turn_cumulative_label_list[index]
        turn_change_label = self.turn_change_label_list[index]
        context = self.context_list[index]
        last_turn_label = self.last_turn_label[index]
        context_mask = self.context_mask_list[index]
        inform_value = self.inform_value_list[index]
        hit_type_dict, hit_value_dict, referred_dict = {}, {}, {}
        for domain_slot in domain_slot_list:
            hit_type_dict[domain_slot] = self.hit_type_list_dict[domain_slot][index]
            hit_value_dict[domain_slot] = self.hit_value_list_dict[domain_slot][index]
            referred_dict[domain_slot] = self.referred_list_dict[domain_slot][index]
        return sample_id, active_domain, active_slot, turn_change_label, turn_cumulative_label, context, context_mask, \
            last_turn_label, hit_type_dict, hit_value_dict, referred_dict, inform_value

    def get_fraction_data(self, fraction):
        assert isinstance(fraction, float) and 0.01 <= fraction <= 1.0
        new_len = math.floor(len(self.sample_id_list) * fraction)
        new_sample_id_list = self.sample_id_list[: new_len]
        new_active_domain_list = self.active_domain_list[: new_len]
        new_active_slot_list = self.active_slot_list[: new_len]
        new_turn_change_label_list = self.turn_change_label_list[: new_len]
        new_turn_cumulative_label_list = self.turn_cumulative_label_list[: new_len]
        new_context_list = self.context_list[: new_len]
        new_context_mask_list = self.context_mask_list[: new_len]
        new_last_turn_label = self.last_turn_label[: new_len]
        new_inform_value = self.inform_value_list[: new_len]
        new_hit_type_list_dict, new_hit_value_list_dict, new_referred_list_dict = {}, {}, {}
        for domain_slot in domain_slot_list:
            new_hit_type_list_dict[domain_slot] = self.hit_type_list_dict[domain_slot][: new_len]
            new_hit_value_list_dict[domain_slot] = self.hit_value_list_dict[domain_slot][: new_len]
            new_referred_list_dict[domain_slot] = self.referred_list_dict[domain_slot][: new_len]
        return new_sample_id_list, new_active_domain_list, new_active_slot_list, new_turn_change_label_list,\
            new_turn_cumulative_label_list, new_context_list, new_context_mask_list, new_last_turn_label,\
            new_hit_type_list_dict, new_hit_value_list_dict, new_referred_list_dict, new_inform_value

    def __len__(self):
        return len(self.sample_id_list)


class Sample(object):
    # check
    def __init__(self, sample_id, active_domain, active_slot, filtered_state, context, context_mask,
                 turn_change_label, turn_cumulative_labels, last_turn_label, inform_value):
        self.sample_id = sample_id
        self.active_domain = active_domain
        self.active_slot = active_slot
        self.turn_change_label = turn_change_label
        self.turn_cumulative_labels = turn_cumulative_labels
        self.last_turn_label = last_turn_label
        self.context = context
        self.context_mask = context_mask
        self.inform_value = inform_value
        self.hit_type, self.hit_value, self.referred,  = {}, {}, {}

        # 注意，上面的turn_change_label和cumulative_label是作为最终evaluation时比较的真值
        # 而filtered state是模型训练时用于优化的y
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


def state_hit_count(data_dict, data_type):
    count_dict = {}
    label_dict = {}
    for domain_slot in domain_slot_list:
        count_dict[domain_slot] = [0, 0]  # for not mention count and valid count
        label_dict[domain_slot] = set()
    for dialogue_idx in data_dict:
        for turn_idx in data_dict[dialogue_idx]:
            state_label = data_dict[dialogue_idx][turn_idx]['turn_cumulative_labels']
            for domain_slot in state_label:
                assert state_label[domain_slot] != ''
                if state_label[domain_slot] == 'none':
                    count_dict[domain_slot][0] += 1
                else:
                    count_dict[domain_slot][1] += 1
                label_dict[domain_slot].add(state_label[domain_slot])
    logger.info('{} label hit count'.format(data_type))
    logger.info(count_dict)
    logger.info('{} label value set'.format(data_type))
    logger.info(label_dict)


def span_case_label_recovery_check(context, context_label_dict, state_dict, dialogue_idx, turn_idx):
    check_flag = True
    for domain_slot in state_dict:
        if domain_slot_type_map[domain_slot] == 'span':
            # 注意，此处的context已经经过截断，所以可能出现无法正常恢复是因为截断的原因，因此如果最后token label为1，则判定可能出现这种情况
            # 并且不视为恢复失败
            end_one_flag = False
            true_label = state_dict[domain_slot].strip()
            context_label = context_label_dict[domain_slot]
            if 1 in context_label:
                start_index = context_label.index(1)
                if 0 in context_label[start_index:]:
                    end_index = context_label[start_index:].index(0) + start_index
                else:
                    end_index = len(context_label)
                    end_one_flag = True
                label_context = context[start_index: end_index]
                reconstruct_label = \
                    tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(label_context)).strip()
                if not approximate_equal_test(reconstruct_label, true_label, variant_flag) and not end_one_flag:
                    check_flag = False
                    print('reconstruct failed, '
                          'reconstruct_label: {}, true_label: {}'.format(reconstruct_label, true_label))
                    print(dialogue_idx)
                    print(turn_idx)
                    print(domain_slot)
    return check_flag


def approximate_equal_test(reconstruct_label, true_label, use_variant):
    reconstruct_label, true_label = reconstruct_label.lower().strip(), true_label.lower().strip()
    if reconstruct_label != true_label:
        if reconstruct_label.replace(' ', '') != true_label.replace(' ', ''):
            if use_variant:
                equal = False
                if reconstruct_label in label_normalize_map:
                    for reconstruct_label_variant in label_normalize_map[reconstruct_label]:
                        reconstruct_true_label = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
                            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + true_label)))).strip()
                        reconstruct_label_variant_ = tokenizer.convert_tokens_to_string(
                            tokenizer.convert_ids_to_tokens(
                                tokenizer.convert_tokens_to_ids(
                                    tokenizer.tokenize(' ' + reconstruct_label_variant)))).strip()
                        trimmed_reconstruct_label = reconstruct_label_variant_.replace(' ', '')
                        trimmed_true_label = reconstruct_true_label.replace(' ', '')
                        if reconstruct_label_variant_ == reconstruct_true_label or trimmed_true_label == \
                                trimmed_reconstruct_label:
                            equal = True
                if true_label in label_normalize_map:
                    for label_variant in label_normalize_map[true_label]:
                        reconstruct_true_label_variant = tokenizer.convert_tokens_to_string(
                            tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(
                                tokenizer.tokenize(' ' + label_variant)))).strip()
                        reconstruct_label_ = tokenizer.convert_tokens_to_string(
                            tokenizer.convert_ids_to_tokens(
                                tokenizer.convert_tokens_to_ids(
                                    tokenizer.tokenize(' ' + reconstruct_label)))).strip()
                        trimmed_reconstruct_label, trimmed_true_label = reconstruct_label_.replace(' ', ''),\
                            reconstruct_true_label_variant.replace(' ', '')
                        if reconstruct_label_ == reconstruct_true_label_variant or \
                                trimmed_true_label == trimmed_reconstruct_label:
                            equal = True
                if not equal:
                    return False
            else:
                return False
    return True


def prepare_data_for_model(data_dict, max_input_length, classify_slot_value_index_dict, interest_domain, data_type):
    data_for_model = []
    for dialogue_idx in data_dict:
        dialogue = data_dict[dialogue_idx]
        for turn_idx in dialogue:
            turn_data = dialogue[turn_idx]
            turn_active_domains = active_domain_structurize(turn_data['active_domain'])
            turn_active_slots = active_slot_structurize(turn_data['active_slots'])
            context_utterance_token_id, context_utterance_token_map_list = \
                tokenize_to_id(turn_data['context_utterance_token'])

            context, context_label_dict, context_mask = alignment_and_truncate(
                context_utterance_token_id, context_utterance_token_map_list, turn_data['state'], max_input_length)
            turn_state = state_structurize(turn_data['state'], context_label_dict, classify_slot_value_index_dict)
            filtered_state = irrelevant_domain_label_mask(turn_state, interest_domain)
            turn_change_label = turn_data['turn_change_state']
            turn_cumulative_labels = turn_data['turn_cumulative_labels']
            turn_inform_value = turn_data['inform_value']
            last_turn_label = turn_data['last_turn_label']

            assert span_case_label_recovery_check(context, context_label_dict, turn_data['turn_cumulative_labels'],
                                                  dialogue_idx, turn_idx)

            data_for_model.append(
                Sample(sample_id=dialogue_idx+'-'+str(turn_idx),
                       active_domain=turn_active_domains,
                       active_slot=turn_active_slots,
                       filtered_state=filtered_state,
                       turn_change_label=turn_change_label,
                       turn_cumulative_labels=turn_cumulative_labels,
                       context=context,
                       last_turn_label=last_turn_label,
                       context_mask=context_mask,
                       inform_value=turn_inform_value)
            )
    # count type
    hit_type_count_list = [0, 0, 0, 0, 0]
    for item in data_for_model:
        for domain_slot in domain_slot_list:
            hit_type = item.hit_type[domain_slot]
            hit_type_count_list[hit_type] += 1
    logger.info('{}, hit_type_count_list: {}'.format(data_type, hit_type_count_list))
    return data_for_model


def irrelevant_domain_label_mask(turn_state, interest_domain):
    # check 用于零次学习，不参与训练
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


def alignment_and_truncate(context_utterance_token_id, context_utterance_token_map_list, turn_state, max_input_length):
    # check
    context_label_dict = {}
    for domain_slot in domain_slot_list:
        token_label = turn_state[domain_slot]['context_token_label']
        aligned_label = []
        assert context_utterance_token_map_list[-1] == len(token_label) - 1
        for origin_index in context_utterance_token_map_list:
            aligned_label.append(token_label[origin_index])
        context_label_dict[domain_slot] = aligned_label

    if len(context_utterance_token_id) > max_input_length:
        context_utterance_token = context_utterance_token_id[: max_input_length]
        for domain_slot in domain_slot_list:
            context_label_dict[domain_slot] = context_label_dict[domain_slot][: max_input_length]
        context_mask = [1 for _ in range(max_input_length)]
    else:
        padding_num = max_input_length - len(context_utterance_token_id)
        padding_token = tokenizer.convert_tokens_to_ids([' '+PAD_token])
        context_mask = [0] * len(context_utterance_token_id) + [1] * padding_num
        context_utterance_token = context_utterance_token_id + padding_token * padding_num
        for domain_slot in domain_slot_list:
            context_label_dict[domain_slot] = context_label_dict[domain_slot] + [0] * padding_num
        assert len(context_mask) == max_input_length

    for domain_slot in domain_slot_list:
        assert len(context_utterance_token) == len(context_label_dict[domain_slot])
    return context_utterance_token, context_label_dict, context_mask


def active_domain_structurize(active_domains):
    # check
    active_domain_list = [0 for _ in range(len(domain_index_map))]
    for domain in active_domains:
        active_domain_list[domain_index_map[domain]] = 1
    return active_domain_list


def active_slot_structurize(active_slots):
    # check
    active_slot_list = [0 for _ in range(len(slot_index_map))]
    for slot in active_slots:
        active_slot_list[slot_index_map[slot]] = 1
    return active_slot_list


def tokenize_to_id(utterance_token):
    origin_token_idx, new_token_list, new_token_origin_token_map_list = 0, list(), list()
    for token in utterance_token:
        new_tokens = tokenizer.tokenize(' ' + token)
        assert len(new_tokens) >= 1
        for new_token in new_tokens:
            new_token_list.append(new_token)
            new_token_origin_token_map_list.append(origin_token_idx)
        origin_token_idx += 1
    new_token_id_list = tokenizer.convert_tokens_to_ids(new_token_list)
    assert len(new_token_list) == len(new_token_id_list)
    return new_token_id_list, new_token_origin_token_map_list


def state_structurize(state, context_label_dict, classify_slot_value_index_dict):
    # check
    # 只有在hit时才会给value index置有效值，其余情况置空，因此前面置的-1之类的值不会产生不恰当的影响
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
    # check
    if no_value_assign == 'miss':
        start_idx, end_idx = -1, -1
    else:
        assert no_value_assign == 'value'
        start_idx, end_idx = 0, 0
    assert domain_slot_type_map[domain_slot] == 'span'
    if 1 in context_label_dict[domain_slot]:
        start_idx = context_label_dict[domain_slot].index(1)
        if 0 not in context_label_dict[domain_slot][start_idx:]:
            end_idx = len(context_label_dict[domain_slot][start_idx:]) + start_idx - 1
        else:
            end_idx = context_label_dict[domain_slot][start_idx:].index(0) + start_idx - 1
    return start_idx, end_idx


def classify_slot_value_indexing(data_dict):
    # checked 注意，classify和none是互斥的，none并不属于合法的classify的一种类别
    classify_slot_value_index_dict, classify_slot_index_value_dict = {}, {}
    for domain_slot in domain_slot_list:
        classify_slot_value_index_dict[domain_slot] = {}
        classify_slot_index_value_dict[domain_slot] = {}
    for dialogue_idx in data_dict:
        for turn in data_dict[dialogue_idx][1]:  # data_dict[dialogue_idx] = [utterance, state, act]
            state_dict, _ = state_extract(None, data_dict[dialogue_idx][1][turn])
            for domain_slot in state_dict:
                if domain_slot_type_map[domain_slot] != 'classify':
                    continue
                classify_value = state_dict[domain_slot]
                if classify_value == 'none' or classify_value == 'dontcare' or \
                        classify_value in classify_slot_value_index_dict[domain_slot]:
                    continue
                idx = len(classify_slot_value_index_dict[domain_slot])
                classify_slot_value_index_dict[domain_slot][classify_value] = idx
                classify_slot_index_value_dict[domain_slot][idx] = classify_value
    return classify_slot_value_index_dict, classify_slot_index_value_dict


def normalize_data(utterance_list, state_dict, act_dict):
    for i in range(len(utterance_list)):
        utterance_list[i] = normalize_text(utterance_list[i])

    return utterance_list, state_dict, act_dict


def dialogue_reorganize_and_normalize(dialogue_idx, utterance_list, state_dict, act_dict,
                                      classify_slot_value_index_dict):
    # check
    reorganize_data = {}
    assert len(utterance_list) % 2 == 0 and dialogue_idx is not None  # dialogue index在此函数中无意义，只是debug时加一个定位参数
    cumulative_labels = {slot: 'none' for slot in domain_slot_list}

    history = ''
    history_token = []
    history_token_label = {}
    for domain_slot in domain_slot_list:
        history_token_label[domain_slot] = []
    for turn_idx in range(0, len(utterance_list)//2):
        reorganize_data[turn_idx] = {}

        active_domain, active_slots, inform_info, inform_slot_filled_dict = \
            act_reorganize_and_normalize(act_dict, turn_idx, args['auxiliary_domain_assign'])
        if turn_idx == 0:
            assert len(active_domain) == 0 and len(active_slots) == 0

        reorganize_data[turn_idx]['active_domain'] = active_domain
        reorganize_data[turn_idx]['active_slots'] = active_slots

        last_turn_label = cumulative_labels.copy()
        modified_slots, cumulative_labels = state_extract(cumulative_labels, state_dict[turn_idx+1])

        system_utterance = normalize_text('' if turn_idx == 0 else utterance_list[2*turn_idx-1].lower())
        if delex_system_utterance:
            system_utterance = delex_text(system_utterance, inform_info)
        user_utterance = normalize_text(utterance_list[2*turn_idx].lower())
        system_utterance_token, user_utterance_token = tokenize(system_utterance), tokenize(user_utterance)
        current_turn_utterance = user_utterance + ' ' + SEP_token + ' ' + system_utterance
        current_turn_utterance_token = user_utterance_token + [SEP_token] + system_utterance_token
        context_utterance = current_turn_utterance + ' ' + SEP_token + ' ' + history
        context_utterance_token = current_turn_utterance_token + [SEP_token] + history_token

        reorganize_data[turn_idx]['turn_utterance'] = CLS_token + ' ' + current_turn_utterance + ' ' + SEP_token
        reorganize_data[turn_idx]['turn_utterance_token'] = [CLS_token] + current_turn_utterance_token + [SEP_token]
        reorganize_data[turn_idx]['history_utterance'] = CLS_token + ' ' + history
        reorganize_data[turn_idx]['history_utterance_token'] = [CLS_token] + history_token
        reorganize_data[turn_idx]['context_utterance'] = CLS_token + ' ' + context_utterance
        reorganize_data[turn_idx]['context_utterance_token'] = [CLS_token] + context_utterance_token

        reorganize_data[turn_idx]['last_turn_label'] = last_turn_label
        reorganize_data[turn_idx]['state'] = {}
        reorganize_data[turn_idx]['inform_value'] = {}
        reorganize_data[turn_idx]['turn_change_state'] = modified_slots.copy()
        reorganize_data[turn_idx]['turn_cumulative_labels'] = cumulative_labels.copy()

        for domain_slot in domain_slot_list:
            reorganize_data[turn_idx]['state'][domain_slot] = {}
            value_label = cumulative_labels[domain_slot]
            inform_label = inform_info[domain_slot] if domain_slot in inform_info else 'none'

            inform_value, referred_slot, token_label, class_type, classify_value = \
                get_turn_label(value_label, inform_label, context_utterance_token,
                               domain_slot, last_turn_label, classify_slot_value_index_dict)
            if class_type == 'unpointable':
                # 同样，unpointable也置空history label token, 这个是label本身的错误
                class_type = 'none'
                referred_slot = 'none'
            else:
                class_type = class_type
                referred_slot = referred_slot

            reorganize_data[turn_idx]['state'][domain_slot]['class_type'] = class_type
            reorganize_data[turn_idx]['state'][domain_slot]['classify_value'] = classify_value
            reorganize_data[turn_idx]['state'][domain_slot]['context_token_label'] = [0] + token_label
            reorganize_data[turn_idx]['state'][domain_slot]['referred_slot'] = referred_slot
            reorganize_data[turn_idx]['inform_value'][domain_slot] = inform_value

        history = context_utterance
        history_token = context_utterance_token
    return reorganize_data


def get_turn_label(value_label, inform_label, context_utterance_token, domain_slot, seen_slots,
                   classify_slot_value_index_dict):
    # four types of class info has it's priority
    # 尽可能提供补充标签
    utterance_token_label = [0 for _ in context_utterance_token]

    informed_value = 'none'
    referred_slot = 'none'
    value_index = -1
    if value_label == 'none' or value_label == 'dontcare':
        class_type = value_label
    else:
        in_utterance_flag, position, value_index = \
            check_label(value_label, context_utterance_token, domain_slot, classify_slot_value_index_dict)
        is_informed, informed_value = check_slot_inform(value_label, inform_label)
        referred_slot = check_slot_referral(value_label, domain_slot, seen_slots)

        if in_utterance_flag:
            # if the slot is referred multi times, use the first time it shows in user utterance
            start_idx, end_idx = position[0]
            for i in range(start_idx, end_idx):
                utterance_token_label[i] = 1

        if is_informed:
            class_type = 'inform'
        elif referred_slot != 'none':
            class_type = 'referral'
        else:
            if domain_slot_type_map[domain_slot] == 'span':
                if in_utterance_flag:
                    class_type = 'hit'
                else:
                    class_type = 'unpointable'
                    unpointable_slot_value_list.append(value_label)
            else:
                assert domain_slot_type_map[domain_slot] == 'classify'
                class_type = 'hit'
    return informed_value, referred_slot, utterance_token_label, class_type, value_index


def check_slot_referral(value_label, slot, seen_slots):
    # check
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
            if approximate_equal_test(seen_slots[s], value_label, variant_flag):
                referred_slot = s
                break
    return referred_slot


def check_label(value_label, user_utterance_token, domain_slot, classify_slot_value_index_dict):
    # check
    in_user_utterance_flag, position, value_index = False, [], -1
    if domain_slot_type_map[domain_slot] == 'classify':
        value_index = classify_slot_value_index_dict[domain_slot][value_label]
    else:
        assert domain_slot_type_map[domain_slot] == 'span'
        in_user_utterance_flag, position = get_token_position(user_utterance_token, value_label)
        # If no hit even though there should be one, check for value label variants
        if (not in_user_utterance_flag) and value_label in label_normalize_map and variant_flag:
            for value_label_variant in label_normalize_map[value_label]:
                in_user_utterance_flag, position = get_token_position(user_utterance_token, value_label_variant)
                if in_user_utterance_flag:
                    break
    return in_user_utterance_flag, position, value_index


def get_token_position(token_list, value_label):
    # check
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
    # check
    # 在做inform时极大程度的放宽了match的判定，只要能够碰上都算，而不是要求严格一致
    result = False
    informed_value = 'none'
    value_label = ' '.join(tokenize(value_label))
    if approximate_equal_test(inform_label, value_label, variant_flag):
        result = True
    if result:
        informed_value = inform_label
    return result, informed_value

#
# def is_in_list(token, value):
#     # check
#     found = False
#     token_list = [item for item in map(str.strip, re.split("(\W+)", token)) if len(item) > 0]
#     value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
#     token_len = len(token_list)
#     value_len = len(value_list)
#     for i in range(token_len + 1 - value_len):  # if the value len is larger than token len, the loop will be skipped
#         if token_list[i:i + value_len] == value_list:
#             found = True
#             break
#     return found


def delex_text(utterance, values, unk_token=UNK_token):
    # check
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
    # check
    utt_lower = normalize_text(utterance)
    utt_tok = [tok for tok in map(str.strip, re.split("(\W+)", utt_lower)) if len(tok) > 0]
    return utt_tok


def state_extract(cumulative_labels, state_dict):
    # checked
    modified_slots = {}
    for domain in state_dict:
        booked = state_dict[domain]['book']['booked']
        booked_slots = {}
        # Check the booked section
        if len(booked) > 0:  # len of booked larger than 0
            for slot in booked[0]:
                booked_slots[slot] = normalize_label('{}-{}'.format(domain, slot), booked[0][slot])  # normalize labels
        # Check the semi and the inform slots
        # 这里的整个逻辑策略是这样，数据集中的state事实上分为两个部分，book和semi
        # book和semi中的结构大抵都是 {key: value}的字典，因此可以如下代码进行赋值
        # 此处有一个特例，book中有一个booked，里面存着的是一个字典再嵌套一个列表。当遍历到booked时，此处的domain_slot会因为不在目标列表中
        # 而booked_slots的信息会在额外的判定中被优先赋值给state，因此这个代码不会有问题
        for category in ['book', 'semi']:
            for slot in state_dict[domain][category]:  # s for slot name
                domain_slot = '{}-book-{}'.format(domain, slot) if category == 'book' else '{}-{}'.format(domain, slot)
                domain_slot = domain_slot.lower()
                value_label = normalize_label(domain_slot, state_dict[domain][category][slot])
                # Prefer the slot value as stored in the booked section
                if slot in booked_slots:
                    value_label = booked_slots[slot]
                # Remember modified slots and entire dialog state
                if cumulative_labels is not None:
                    if domain_slot in domain_slot_list and cumulative_labels[domain_slot] != value_label:
                        modified_slots[domain_slot] = value_label
                        cumulative_labels[domain_slot] = value_label
                else:
                    if domain_slot in domain_slot_list:
                        modified_slots[domain_slot] = value_label
    if cumulative_labels is not None:
        return modified_slots.copy(), cumulative_labels.copy()
    else:
        return modified_slots.copy(), None


def act_reorganize_and_normalize(act_dict, turn_idx, auxiliary_domain_assign):
    """
    check
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
            else:
                if slot not in UNNORMALIZED_ACTION_SLOT:
                    print(slot)

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
                if domain_slot not in inform_info and domain_slot in domain_slot_list:
                    inform_info[domain_slot] = normalize_label(domain_slot, value)

    # overwrite act if it has booking value
    # 如果出现了booking打头的情况，则将所有可能符合的label全部置1
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
                domain_slot = (act_domain + '-' + slot).strip()
                if domain_slot in ACT_MAP_DICT:
                    domain_slot = ACT_MAP_DICT[domain_slot].strip()
                if len(domain_slot.split('-')) != 3:
                    continue
                domain_slots = []
                slot = domain_slot.split('-')[1] + domain_slot.split('-')[2]
                for name in NORMALIZE_MAP['slots']:
                    if len(active_domain) > 0:
                        if name.__contains__(slot):
                            domain_slots.append(name)
                    else:
                        if name.__contains__(slot):
                            domain_slots.append(name)

                # If the booking slot is already filled skip
                for domain_slot in domain_slots:
                    if domain_slot not in inform_info and domain_slot in domain_slot_list:
                        inform_info[domain_slot] = normalize_label(domain_slot, value)

    for domain_slot in domain_slot_list:
        if domain_slot in inform_info:
            inform_slot_filled_dict[domain_slot] = 1
    return active_domain, active_slots, inform_info, inform_slot_filled_dict


def assign_domain_when_booking(act_dict, turn_idx):
    # check
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
    # checked, 注意，act和state的turn idx从1算起， 此处载入数据的state时没有做任何数据预处理工作
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
    # check
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
    logger.info('length of train dialogue: {}, length of dev dialogue: {}, length of test dialogue: {}'
                .format(len(train_idx_list), len(dev_idx_list), len(test_idx_list)))
    return train_idx_list, dev_idx_list, test_idx_list


def normalize_time(text):
    # checked
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
    # copy
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
def normalize_label(domain_slot, value_label):
    # checked
    # 根据设计，不同的slot对应不同的标准化方法（比如时间和其他的就不一样），因此要输入具体的slot name
    # Normalization of empty slots
    if isinstance(value_label, str):
        value_label = value_label.strip().lower()
    if value_label == '' or value_label == "not mentioned":
        return "none"

    if value_label == 'dontcare' or value_label == 'dont care' or value_label == 'don\'t care' or \
            value_label == 'doesn\'t care':
        return "dontcare"

    # Normalization of time slots
    if "leaveat" in domain_slot or "arriveby" in domain_slot or domain_slot == 'restaurant-book-time':
        return normalize_time(value_label)

    # Normalization
    if "type" in domain_slot or "name" in domain_slot or "destination" in domain_slot or "departure" in domain_slot:
        value_label = re.sub("guesthouse", "guest house", value_label)

    if domain_slot == 'restaurant-book-day' or domain_slot == 'hotel-book-day' or domain_slot == 'train-day':
        if value_label == 'thur':
            value_label = 'thursday'
        if value_label == 'w' or value_label == 'we':
            value_label = 'wednesday'
        if value_label == 'not given':
            value_label = 'none'
        if value_label == 'monda':
            value_label = 'monday'
        if value_label == 'next friday' or value_label == 'fr':
            value_label = 'friday'
        if value_label == 'n':
            value_label = 'none'

    if domain_slot == 'restaurant-pricerange' or domain_slot == 'hotel-pricerange':
        if value_label == 'mode' or value_label == 'mo' or value_label == 'moderately':
            value_label = 'moderate'
        if value_label == 'ch' or value_label == 'inexpensive':
            value_label = 'cheap'
        if value_label == 'any':
            value_label = 'dontcare'
        if value_label == 'not':
            value_label = 'none'

    if domain_slot == 'hotel-parking':
        if value_label == 'y' or value_label == 'free' or value_label == 'free parking':
            value_label = 'yes'
        if value_label == 'n':
            value_label = 'no'
    if domain_slot == 'hotel-book-people' or domain_slot == 'hotel-book-stay':
        if value_label == 'six':
            value_label = '6'
        if value_label == '3.':
            value_label = '3'
    if domain_slot == 'hotel-internet':
        if value_label == 'free' or value_label == 'free internet' or value_label == 'y':
            value_label = 'yes'
        if value_label == 'does not':
            value_label = 'no'
    if domain_slot == 'train-book-people':
        if value_label == '`1':
            value_label = '1'
    if domain_slot == 'hotel-stars':
        if value_label == 'four star' or value_label == 'four stars' or value_label == '4-star' or \
                value_label == '4 star':
            value_label = '4'
        if value_label == 'two':
            value_label = '2'
        if value_label == 'three':
            value_label = '3'
    if domain_slot == 'attraction-type':
        if value_label == 'mutiple sports' or value_label == 'mutliple sports':
            value_label = 'multiple sports'
        if value_label == 'swimmingpool' or value_label == 'pool':
            value_label = 'swimming pool'
        if value_label == 'concerthall' or value_label == 'concert':
            value_label = 'concert hall'
        if value_label == 'night club':
            value_label = 'nightclub'
        if value_label == 'colleges' or value_label == 'coll':
            value_label = 'college'
        if value_label == 'architectural':
            value_label = 'architecture'
        if value_label == 'mus':
            value_label = 'museum'
        if value_label == 'galleria':
            value_label = 'gallery'
    return value_label


def main():
    logger.info('label_map load success')
    data, classify_slot_value_index_dict, classify_slot_index_value_dict = prepare_data(True)
    train_loader, dev_loader, test_loader = data
    print(len(train_loader))
    batch_count = 0
    for _ in tqdm(train_loader):
        batch_count += 1
    for _ in tqdm(dev_loader):
        batch_count += 1
    for _ in tqdm(test_loader):
        batch_count += 1
    print(batch_count)
    logger.info('data read success')
    unpointable_slot_value_set = set(unpointable_slot_value_list)
    print(unpointable_slot_value_set)
    print('unpointable_count: {}'.format(len(unpointable_slot_value_list)))


if __name__ == '__main__':
    main()
