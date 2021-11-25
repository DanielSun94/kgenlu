import json
import os
import pickle
from tqdm import tqdm
import re
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from history_config import logger, args, preprocessed_cache_path, label_normalize_path, dialogue_data_path, \
    test_idx_path, dev_idx_path, act_data_path, PAD_token, SEP_token, CLS_token, UNK_token
from kgenlu_read_data import get_classify_slot_index_map, get_dataset_idx, SampleDataset, get_dialogue_info, \
    act_reorganize_and_normalize, normalize_text, delex_text, tokenize, check_label, check_slot_inform


NORMALIZE_MAP = json.load(open(label_normalize_path, 'r'))
domain_slot_list, domain_index_map, slot_index_map, domain_slot_type_map, label_normalize_map = \
    NORMALIZE_MAP['slots'], NORMALIZE_MAP['domain_index'], NORMALIZE_MAP['slot_index'], NORMALIZE_MAP['slots-type'],\
    NORMALIZE_MAP['label_maps']
train_domain, test_domain, use_history_label, use_history_utterance, use_multiple_gpu, train_data_fraction, \
    dev_data_fraction, test_data_fraction, no_value_assign_strategy, max_len, delex_system_utterance = \
    args['train_domain'], args['test_domain'], args['use_history_label'], args['use_history_utterance'], \
    args['multi_gpu'], args['train_data_fraction'], args['dev_data_fraction'], args['test_data_fraction'], \
    args['no_value_assign_strategy'],  args['max_len'], args['delex_system_utterance']
train_domain_set = set(train_domain.strip().split('$'))
test_domain_set = set(test_domain.strip().split('$'))
overwrite = args['overwrite_cache']


def prepare_data():
    if os.path.exists(preprocessed_cache_path) and not overwrite:
        train_data, dev_data, test_data, classify_slot_value_index_dict, classify_slot_index_value_dict = \
            pickle.load(open(preprocessed_cache_path, 'rb'))
    else:
        train_idx_list, dev_idx_list, test_idx_list = get_dataset_idx(dev_idx_path, test_idx_path, dialogue_data_path)
        idx_dict = {'train': train_idx_list, 'dev': dev_idx_list, 'test': test_idx_list}
        dialogue_data = json.load(open(dialogue_data_path, 'r'))
        act_data = json.load(open(act_data_path, 'r'))
        classify_slot_value_index_dict, classify_slot_index_value_dict = \
            get_classify_slot_index_map(idx_dict, dialogue_data, act_data)
        dev_data = process_data(idx_dict['dev'], dialogue_data, act_data, 'dev', classify_slot_value_index_dict)
        test_data = process_data(idx_dict['test'], dialogue_data, act_data, 'test', classify_slot_value_index_dict)
        train_data = process_data(idx_dict['train'], dialogue_data, act_data, 'train', classify_slot_value_index_dict)
        pickle.dump((train_data, dev_data, test_data, classify_slot_value_index_dict,
                     classify_slot_index_value_dict), open(preprocessed_cache_path, 'wb'))
    logger.info('data preprocessed')

    assert 0.01 <= float(train_data_fraction) <= 1 and 0.01 <= float(dev_data_fraction) <= 1 \
           and 0.01 <= float(test_data_fraction) <= 1
    train_data = SampleDataset(*train_data.get_fraction_data(float(train_data_fraction)))
    dev_data = SampleDataset(*dev_data.get_fraction_data(float(dev_data_fraction)))
    test_data = SampleDataset(*test_data.get_fraction_data(float(test_data_fraction)))
    if use_multiple_gpu:
        train_sampler, dev_sampler, test_sampler = \
            DistributedSampler(train_data), DistributedSampler(dev_data), DistributedSampler(test_data)
    else:
        train_sampler, dev_sampler, test_sampler = \
            RandomSampler(train_data), RandomSampler(dev_data), RandomSampler(test_data)
    # train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args['batch_size'], collate_fn=collate_fn)
    # dev_loader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args['batch_size'], collate_fn=collate_fn)
    # test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=args['batch_size'], collate_fn=collate_fn)
    # logger.info('data prepared')
    # return train_loader, dev_loader, test_loader, classify_slot_value_index_dict, classify_slot_index_value_dict


def process_data(idx_list, dialogue_dict, act, data_type, class_slot_value_index_dict):
    data_dict, raw_data_dict, idx_set = {}, {}, set(idx_list)
    for dialogue_idx in dialogue_dict:
        if dialogue_idx not in idx_set:
            continue
        if dialogue_idx.strip().split('.')[0] not in act:
            logger.info('dialog act of {} not found'.format(dialogue_idx))
        raw_data_dict[dialogue_idx] = get_dialogue_info(act, dialogue_dict, dialogue_idx)

    for dialogue_idx in raw_data_dict:
        utterance_list, state_dict, act_dict = raw_data_dict[dialogue_idx]
        data_dict[dialogue_idx] = data_reorganize(utterance_list, state_dict, act_dict, class_slot_value_index_dict)

    # if data_type == 'train' or 'dev':
    #     processed_data = prepare_data_for_model(data_dict, max_len, class_slot_value_index_dict,
    #                                             train_domain_set)
    # else:
    #     assert data_type == 'test'
    #     processed_data = prepare_data_for_model(data_dict, max_len, class_slot_value_index_dict, test_domain_set)
    # logger.info('prepare process finished')
    # logger.info('constructing dataloader')
    # dataset = construct_dataloader(processed_data, data_type)
    # return dataset


def data_reorganize(utterance_list, state_dict, act_dict, class_slot_value_index_dict):
    # check
    reorganize_data = {}
    assert len(utterance_list) % 2 == 0
    # 与原始版本相比，不仅需要知道值，还需要知道嵌入轮数。如果一个mention是在inform之后被计入到state中，则单独记两次
    # item structure
    # {'no': num_idx, 'turn_index': turn_index, 'mentioned_type': 'informed' or 'in state', 'domain': domain,
    # 'slot': slot, 'value': value}
    seen_slots = []

    history, history_token, history_token_label = '', [], {}
    for domain_slot in domain_slot_list:
        history_token_label[domain_slot] = []

    for turn_idx in range(0, len(utterance_list) // 2):
        reorganize_data[turn_idx] = {}

        # reorganize and set act information
        active_domain, active_slots, inform_info, inform_slot_filled_dict = \
            act_reorganize_and_normalize(act_dict, turn_idx, args['auxiliary_domain_assign'])
        if turn_idx == 0:
            assert len(active_domain) == 0 and len(active_slots) == 0
        reorganize_data[turn_idx]['active_domain'] = active_domain
        reorganize_data[turn_idx]['active_slots'] = active_slots

        # reorganize and set utterance info
        system_utterance = normalize_text('' if turn_idx == 0 else utterance_list[2 * turn_idx - 1].lower())
        if delex_system_utterance:
            system_utterance = delex_text(system_utterance, inform_info)
        user_utterance = normalize_text(utterance_list[2 * turn_idx].lower())
        system_utterance_token, user_utterance_token = tokenize(system_utterance), tokenize(user_utterance)
        current_turn_utterance = user_utterance + ' ' + SEP_token + ' ' + system_utterance
        current_turn_utterance_token = user_utterance_token + [SEP_token] + system_utterance_token
        reorganize_data[turn_idx]['turn_utterance'] = CLS_token + ' ' + current_turn_utterance + ' ' + SEP_token
        reorganize_data[turn_idx]['turn_utterance_token'] = [CLS_token] + current_turn_utterance_token + [SEP_token]
        reorganize_data[turn_idx]['history_utterance'] = CLS_token + ' ' + history
        reorganize_data[turn_idx]['history_utterance_token'] = [CLS_token] + history_token

        # modified_slots代表本轮对话后的需要预测的label，其仅仅包含了本轮对话中发生改变的slot value
        # mentioned_slots代表了上一轮的seen slots和本轮的inform信息综合后的结果
        modified_slots, mentioned_slots, seen_slots = state_extract(seen_slots, state_dict[turn_idx + 1])
        reorganize_data[turn_idx]['state'] = {}
        reorganize_data[turn_idx]['inform_value'] = {}

        # reorganize and set state info
        for domain_slot in domain_slot_list:
            reorganize_data[turn_idx]['state'][domain_slot] = {}
            value_label = 'none'
            if domain_slot in modified_slots:
                value_label = modified_slots[domain_slot]
            inform_label = inform_info[domain_slot] if domain_slot in inform_info else 'none'

            inform_value, turn_utterance_token_label, class_type, classify_value = \
                get_turn_label(value_label, inform_label, current_turn_utterance_token,
                               domain_slot, seen_slots, class_slot_value_index_dict)

            assert len(turn_utterance_token_label) == len(system_utterance_token) + len(user_utterance_token) + 1

            if class_type == 'unpointable':
                class_type = 'none'
            else:
                class_type = class_type

            reorganize_data[turn_idx]['state'][domain_slot]['class_type'] = class_type
            reorganize_data[turn_idx]['state'][domain_slot]['classify_value'] = classify_value
            reorganize_data[turn_idx]['state'][domain_slot]['token_label'] = [0] + turn_utterance_token_label + [0]
            reorganize_data[turn_idx]['state'][domain_slot]['history_token_label'] = \
                [0] + history_token_label[domain_slot]
            reorganize_data[turn_idx]['change_state'] = modified_slots
            reorganize_data[turn_idx]['cumulative_labels'] = cumulative_labels

            # update history token label
            assert len(history_token_label[domain_slot]) == len(history_token)
            history_token_label[domain_slot] = turn_utterance_token_label + [0] + history_token_label[domain_slot]
        history = current_turn_utterance + ' ' + SEP_token + ' ' + history
        history_token = current_turn_utterance_token + [SEP_token] + history_token
    return reorganize_data


def state_extract(cumulative_labels, state_dict):
    # TODO
    return 0, 0


def get_turn_label(value_label, inform_label, current_turn_utterance_token, domain_slot, cumulative_labels,
                   class_slot_value_index_dict):
    utterance_token_label = [0 for _ in current_turn_utterance_token]

    informed_value = 'none'
    value_index = -1
    if value_label == 'none' or value_label == 'dontcare':
        class_type = class_slot_value_index_dict[domain_slot][value_label]
        is_mentioned = False
        mentioned_info = []
    else:
        in_utterance_flag, position, value_index = check_label(value_label, current_turn_utterance_token, domain_slot,
                                                               class_slot_value_index_dict)
        is_mentioned, mentioned_info = label_mentioned(value_label, domain_slot, inform_label, cumulative_labels)

        if in_utterance_flag:
            # if the slot is referred multi times, use the first time it shows in user utterance
            start_idx, end_idx = position[0]
            for i in range(start_idx, end_idx):
                utterance_token_label[i] = 1

        if domain_slot_type_map[domain_slot] == 'span':
             if in_utterance_flag:
                class_type = 'hit'
             else:
                class_type = 'unpointable'
        else:
            assert domain_slot_type_map[domain_slot] == 'classify'
            class_type = 'hit'
    return class_type, utterance_token_label, value_index, is_mentioned, mentioned_info


def label_mentioned(value_label, domain_slot, inform_label, cumulative_labels):
    is_mentioned = False
    mentioned_info = {'type': 'previous_label', 'domain': '', 'slot': '', 'turn_difference': ''}
    return is_mentioned, mentioned_info


def main():
    train_loader, dev_loader, test_loader, class_slot_value_index_dict, class_slot_index_value_dict = prepare_data()
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


if __name__ == "__main__":
    main()
