import os
import pickle
import json
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from history_config import args, dev_idx_path, test_idx_path, act_data_path, label_normalize_path,\
    dialogue_data_path, SEP_token, CLS_token, cache_path, logger, MENTIONED_MAP_LIST
from transformers import RobertaTokenizer
import random
import torch
# use previous code
from base_read_data import get_classify_slot_index_map, span_case_label_recovery_check, approximate_equal_test, \
    get_dataset_idx, get_dialogue_info, SampleDataset, normalize_text, state_extract, \
    delex_text, act_reorganize_and_normalize, tokenize, normalize_label, check_label, active_slot_structurize, \
    active_domain_structurize, tokenize_to_id, alignment_and_truncate, span_idx_extract


if 'roberta' in args['pretrained_model']:
    tokenizer = RobertaTokenizer.from_pretrained(args['pretrained_model'])
else:
    raise ValueError('')

NORMALIZE_MAP = json.load(open(label_normalize_path, 'r'))
dialogue_data = json.load(open(dialogue_data_path, 'r'))
act_data = json.load(open(act_data_path, 'r'))

overwrite_cache = args['overwrite_cache']
use_multiple_gpu = args['multi_gpu']
train_data_fraction = args['train_data_fraction']
no_value_assign_strategy = args['no_value_assign_strategy']
max_len = args['max_len']
delex_system_utterance = args['delex_system_utterance']
variant_flag = args['use_label_variant']
train_domain = args['train_domain']
test_domain = args['test_domain']
train_domain_set = set(train_domain.strip().split('$'))
test_domain_set = set(test_domain.strip().split('$'))

domain_slot_list = NORMALIZE_MAP['slots']
domain_index_map = NORMALIZE_MAP['domain_index']
slot_index_map = NORMALIZE_MAP['slot_index']
domain_slot_type_map = NORMALIZE_MAP['slots-type']
label_normalize_map = NORMALIZE_MAP['label_maps']

active_slot_count = dict()
unpointable_slot_value_list = []


def main():
    logger.info('label_map load success')
    _, __, train_loader, dev_loader, test_loader = prepare_data()
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


def prepare_data():
    logger.info('start loading data, overwrite flag: {}'.format(overwrite_cache))
    if os.path.exists(cache_path) and (not overwrite_cache):
        classify_slot_value_index_dict, classify_slot_index_value_dict, train_data, dev_data, test_data = \
            pickle.load(open(cache_path, 'wb'))
    else:
        train_idx_list, dev_idx_list, test_idx_list = get_dataset_idx(dev_idx_path, test_idx_path, dialogue_data_path)
        idx_dict = {'train': train_idx_list, 'dev': dev_idx_list, 'test': test_idx_list}
        classify_slot_value_index_dict, classify_slot_index_value_dict = \
            get_classify_slot_index_map(idx_dict, dialogue_data, act_data)
        dev_data = process_data(idx_dict['dev'], dialogue_data, act_data, 'dev', classify_slot_value_index_dict)
        dev_data = construct_dataloader(dev_data, 'dev')
        test_data = process_data(idx_dict['test'], dialogue_data, act_data, 'test', classify_slot_value_index_dict)
        test_data = construct_dataloader(test_data, 'test')
        train_data = process_data(idx_dict['train'], dialogue_data, act_data, 'train', classify_slot_value_index_dict)
        train_data = construct_dataloader(train_data, 'train')
        pickle.dump([classify_slot_value_index_dict, classify_slot_index_value_dict, train_data, dev_data, test_data],
                    open(cache_path, 'wb'))
    logger.info('data loaded')
    logger.info('constructing dataloader')

    assert 0.01 <= float(train_data_fraction) <= 1
    train_data = SampleDataset(*train_data.get_fraction_data(float(train_data_fraction)))
    dev_sampler, test_sampler = SequentialSampler(dev_data), SequentialSampler(dev_data)
    train_sampler = DistributedSampler(train_data) if use_multiple_gpu else RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args['batch_size'], collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args['batch_size'], collate_fn=collate_fn)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=args['batch_size'], collate_fn=collate_fn)
    return classify_slot_value_index_dict, classify_slot_index_value_dict, train_loader, dev_loader, test_loader


def construct_dataloader(processed_data, data_type):
    # check
    sample_id_list = [item.sample_id for item in processed_data]
    active_domain_list = [item.active_domain for item in processed_data]
    active_slot_list = [item.active_slot for item in processed_data]
    context_list = [item.context for item in processed_data]
    context_mask_list = [item.context_mask for item in processed_data]
    mentioned_slot_list = [item.mentioned_slot_list for item in processed_data]
    label_list_dict, hit_type_list_dict, mentioned_idx_list_dict, hit_value_list_dict = {}, {}, {}, {}
    for domain_slot in domain_slot_list:
        label_list_dict[domain_slot] = [item.label[domain_slot] for item in processed_data]
        hit_type_list_dict[domain_slot] = [item.hit_type[domain_slot] for item in processed_data]
        mentioned_idx_list_dict[domain_slot] = [item.mentioned_idx[domain_slot] for item in processed_data]
        hit_value_list_dict[domain_slot] = [item.hit_value[domain_slot] for item in processed_data]

    # 此处由于train可以默认为知道上一轮结果真值，因此可以shuffle。而dev和test不知道，需要依赖预测进行判断，因此必须
    if data_type == 'train':
        idx_list = [i for i in range(len(sample_id_list))]
        random.shuffle(idx_list)
        new_sample_id_list, new_active_domain_list, new_active_slot_list, new_context_list, new_context_mask_list, \
            new_mentioned_slot_list, new_label_list_dict, new_hit_type_list_dict, new_mentioned_idx_list_dict,\
            new_hit_value_list_dict = [], [], [], [], [], [], {}, {}, {}, {}
        for domain_slot in domain_slot_list:
            new_label_list_dict[domain_slot] = []
            new_hit_type_list_dict[domain_slot] = []
            new_mentioned_idx_list_dict[domain_slot] = []
            new_hit_value_list_dict[domain_slot] = []
        for idx in idx_list:
            new_sample_id_list.append(sample_id_list[idx])
            new_active_domain_list.append(active_domain_list[idx])
            new_active_slot_list.append(active_slot_list[idx])
            new_context_list.append(context_list[idx])
            new_context_mask_list.append(context_mask_list[idx])
            new_mentioned_slot_list.append(mentioned_slot_list[idx])
            for domain_slot in domain_slot_list:
                new_hit_type_list_dict[domain_slot].append(hit_type_list_dict[domain_slot][idx])
                new_hit_value_list_dict[domain_slot].append(hit_value_list_dict[domain_slot][idx])
                new_mentioned_idx_list_dict[domain_slot].append(mentioned_idx_list_dict[domain_slot][idx])
                new_label_list_dict[domain_slot].append(label_list_dict[domain_slot][idx])
        dataset = \
            SampleDataset(new_sample_id_list, new_active_domain_list, new_active_slot_list, new_context_list,
                          new_context_mask_list, new_mentioned_slot_list, new_label_list_dict, new_hit_type_list_dict,
                          new_mentioned_idx_list_dict, new_hit_value_list_dict)
    else:
        dataset = SampleDataset(sample_id_list, active_domain_list, active_slot_list, context_list, context_mask_list,
                                mentioned_slot_list, label_list_dict, hit_type_list_dict, mentioned_idx_list_dict,
                                hit_value_list_dict)
    return dataset


class SampleDataset(Dataset):
    def __init__(self, sample_id_list, active_domain_list, active_slot_list, context_list, context_mask_list,
                 mentioned_slot_list, label_list_dict, hit_type_list_dict, mentioned_idx_list_dict,
                 hit_value_list_dict):
        self.sample_id_list = sample_id_list
        self.active_domain_list = active_domain_list
        self.active_slot_list = active_slot_list
        self.context_list = context_list
        self.context_mask_list = context_mask_list
        self.mentioned_slot_list = mentioned_slot_list
        self.label_list_dict = label_list_dict
        self.hit_type_list_dict = hit_type_list_dict
        self.mentioned_idx_list_dict = mentioned_idx_list_dict
        self.hit_value_list_dict = hit_value_list_dict

    def __getitem__(self, index):
        sample_id = self.sample_id_list[index]
        active_domain = self.active_domain_list[index]
        active_slot = self.active_slot_list[index]
        context = self.context_list[index]
        context_mask = self.context_mask_list[index]
        mentioned_slot = self.mentioned_slot_list[index]
        hit_type_dict, hit_value_dict, label_dict, mentioned_idx_dict = {}, {}, {}, {}
        for domain_slot in domain_slot_list:
            hit_type_dict[domain_slot] = self.hit_type_list_dict[domain_slot][index]
            hit_value_dict[domain_slot] = self.hit_value_list_dict[domain_slot][index]
            label_dict[domain_slot] = self.label_list_dict[domain_slot][index]
            mentioned_idx_dict[domain_slot] = self.mentioned_idx_list_dict[domain_slot][index]
        return sample_id, active_domain, active_slot, context, context_mask, label_dict, hit_type_dict, mentioned_slot,\
            hit_value_dict

    def get_fraction_data(self, fraction):
        assert isinstance(fraction, float) and 0.01 <= fraction <= 1.0
        new_len = math.floor(len(self.sample_id_list) * fraction)
        new_sample_id_list = self.sample_id_list[: new_len]
        new_active_domain_list = self.active_domain_list[: new_len]
        new_active_slot_list = self.active_slot_list[: new_len]
        new_context_list = self.context_list[: new_len]
        new_context_mask_list = self.context_mask_list[: new_len]
        new_mentioned_slot_list = self.mentioned_slot_list[: new_len]
        new_hit_type_list_dict, new_hit_value_list_dict, new_label_list_dict, new_mentioned_idx_list_dict = \
            {}, {}, {}, {}
        for domain_slot in domain_slot_list:
            new_hit_type_list_dict[domain_slot] = self.hit_type_list_dict[domain_slot][: new_len]
            new_hit_value_list_dict[domain_slot] = self.hit_value_list_dict[domain_slot][: new_len]
            new_label_list_dict[domain_slot] = self.label_list_dict[domain_slot][: new_len]
            new_mentioned_idx_list_dict[domain_slot] = self.mentioned_idx_list_dict[domain_slot][: new_len]
        return new_sample_id_list, new_active_domain_list, new_active_slot_list, new_context_list, \
            new_context_mask_list, new_mentioned_slot_list, new_label_list_dict, new_hit_type_list_dict, \
            new_mentioned_idx_list_dict, new_hit_value_list_dict

    def __len__(self):
        return len(self.sample_id_list)


def collate_fn(batch):
    sample_id_list, active_domain_list, active_slot_list, context_list, context_mask_list, mentioned_slot_list, \
        label_list_dict, hit_type_list_dict, mentioned_idx_list_dict, hit_value_list_dict = \
        [], [], [], [], [], [], {}, {}, {}, {}
    for domain_slot in domain_slot_list:
        label_list_dict[domain_slot] = []
        hit_type_list_dict[domain_slot] = []
        hit_value_list_dict[domain_slot] = []
        mentioned_idx_list_dict[domain_slot] = []
    for sample in batch:
        sample_id_list.append(sample[0])
        active_domain_list.append(sample[1])
        active_slot_list.append(sample[2])
        context_list.append(sample[3])
        context_mask_list.append(sample[4])
        mentioned_slot_list.append(sample[5])
        for domain_slot in domain_slot_list:
            label_list_dict[domain_slot].append(sample[6][domain_slot])
            hit_type_list_dict[domain_slot].append(sample[7][domain_slot])
            hit_value_list_dict[domain_slot].append(sample[8][domain_slot])
            mentioned_idx_list_dict[domain_slot].append(sample[9][domain_slot])

    active_domain_list = torch.FloatTensor(active_domain_list)
    active_slot_list = torch.FloatTensor(active_slot_list)
    context_list = torch.LongTensor(context_list)
    context_mask_list = torch.BoolTensor(context_mask_list)
    for domain_slot in domain_slot_list:
        hit_type_list_dict[domain_slot] = torch.LongTensor(hit_type_list_dict[domain_slot])
        hit_value_list_dict[domain_slot] = torch.LongTensor(hit_value_list_dict[domain_slot])
        mentioned_idx_list_dict[domain_slot] = torch.LongTensor(mentioned_idx_list_dict[domain_slot])
    return sample_id_list, active_domain_list, active_slot_list, context_list, context_mask_list, mentioned_slot_list, \
        label_list_dict, hit_type_list_dict, mentioned_idx_list_dict, hit_value_list_dict


def process_data(idx_list, dialogue_dict, act, data_type, classify_slot_value_index_dict):
    data_dict, raw_data_dict, idx_set = {}, {}, set(idx_list)

    for idx in dialogue_dict:
        if idx not in idx_set:
            continue
        if idx.strip().split('.')[0] not in act:
            logger.info('act of {} not found'.format(idx))
        utterance_list, state_dict, act_dict = get_dialogue_info(act, dialogue_dict, idx)
        data_dict[idx] = dialogue_reorganize(idx, utterance_list, state_dict, act_dict, classify_slot_value_index_dict)

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
    return processed_data


def state_hit_count(data_dict, data_type):
    count_dict, label_dict = {}, {}
    for domain_slot in domain_slot_list:
        count_dict[domain_slot], label_dict[domain_slot] = [0, 0], set()  # for not mention count and valid count
    for dialogue_idx in data_dict:
        for turn_idx in data_dict[dialogue_idx]:
            state_label = data_dict[dialogue_idx][turn_idx]['label']
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


def prepare_data_for_model(data_dict, max_input_length, class_slot_value_index_dict, interest_domain, data_type):
    data_for_model = []
    for dialogue_idx in data_dict:
        dialogue = data_dict[dialogue_idx]
        for turn_idx in dialogue:
            data = dialogue[turn_idx]
            active_domain = active_domain_structurize(data['active_domain'])
            active_slot = active_slot_structurize(data['active_slot'])
            utterance_token_id, utterance_token_map_list = tokenize_to_id(data['context_utterance_token'])
            mentioned_slot_split_list = [(0, 0, 0, 0, 0)]
            mentioned_slot_list = data['mentioned_slot_list']
            state = data['state']
            label = data['label']

            for item in mentioned_slot_list:
                turn_idx, mentioned_type, domain, slot, value = item.split('$')
                turn_id = tokenizer.convert_tokens_to_ids([" "+i for i in tokenize(" "+turn_idx)])
                domain_id = tokenizer.convert_tokens_to_ids([" "+i for i in tokenize(" " + domain)])
                slot_id = tokenizer.convert_tokens_to_ids([" "+i for i in tokenize(" " + slot.replace('book-', ''))])
                value_id = tokenizer.convert_tokens_to_ids([" "+i for i in tokenize(" " + value)])
                mentioned_id = tokenizer.convert_tokens_to_ids([" "+i for i in tokenize(" " + mentioned_type)])
                mentioned_slot_split_list.append(turn_id+domain_id+slot_id+value_id+mentioned_id)

            context, context_label_dict, context_mask = alignment_and_truncate(
                utterance_token_id, utterance_token_map_list, data['state'], max_input_length)
            turn_state = state_structurize(state, mentioned_slot_list, context_label_dict,
                                           class_slot_value_index_dict)
            filtered_state = irrelevant_domain_label_mask(turn_state, interest_domain)

            assert span_case_label_recovery_check(context, context_label_dict, label, dialogue_idx, turn_idx)
            data_for_model.append(DataSample(
                sample_id=dialogue_idx+'-'+str(turn_idx), active_domain=active_domain, active_slot=active_slot,
                filtered_state=filtered_state, label=label, context=context, context_mask=context_mask,
                mentioned_slot_list=mentioned_slot_split_list))
    # count type
    hit_type_count_list = [0, 0, 0, 0]
    for item in data_for_model:
        for domain_slot in domain_slot_list:
            hit_type = item.hit_type[domain_slot]
            hit_type_count_list[hit_type] += 1
    logger.info('{}, hit_type_count_list: {}'.format(data_type, hit_type_count_list))
    return data_for_model


def state_structurize(state, mentioned_slots, context_label_dict, classify_slot_value_index_dict):
    # check
    # 只有在hit时才会给value index置有效值，其余情况置空，因此前面置的-1之类的值不会产生不恰当的影响
    reorganized_state = {}
    for domain_slot in state:
        if domain_slot not in reorganized_state:
            reorganized_state[domain_slot] = {}
        class_type = state[domain_slot]['class_type']
        classify_value_index = state[domain_slot]['classify_value']

        # initialize
        if no_value_assign_strategy == 'miss':
            mentioned_idx = -1
            if domain_slot_type_map[domain_slot] == 'classify':
                hit_value = -1
            else:
                hit_value = -1, -1
        elif no_value_assign_strategy == 'value':
            if domain_slot_type_map[domain_slot] == 'classify':
                hit_value = len(classify_slot_value_index_dict[domain_slot])
            else:
                hit_value = 0, 0
            mentioned_idx = 0
        else:
            raise ValueError('')

        # class_type_label
        if class_type == 'none' or class_type == 'unpointable':
            hit_type = 0
        elif class_type == 'dontcare':
            hit_type = 1
        elif class_type == 'mentioned':
            hit_type = 2
            mentioned_slot = state[domain_slot]['mentioned_slot']
            mentioned_idx = mentioned_slots.index(mentioned_slot) + 1
        else:
            assert class_type == 'hit'
            hit_type = 3
            if domain_slot_type_map[domain_slot] == 'classify':
                hit_value = classify_value_index

        # 注意，因为history的补充label，因此span case的标签标记是和class type无关，终归是要做的
        if domain_slot_type_map[domain_slot] == 'span':
            hit_value = span_idx_extract(context_label_dict, domain_slot, no_value_assign_strategy)

        reorganized_state[domain_slot]['hit_value'] = hit_value
        reorganized_state[domain_slot]['hit_type'] = hit_type
        reorganized_state[domain_slot]['mentioned_idx'] = mentioned_idx
    return reorganized_state


def irrelevant_domain_label_mask(turn_state, interest_domain):
    # check 用于零次学习，不参与训练
    filtered_turn_state = {}
    for domain_slot in domain_slot_list:
        filtered_turn_state[domain_slot] = {}
        if domain_slot.strip().split('-')[0] not in interest_domain:
            filtered_turn_state[domain_slot]['hit_type'] = -1
            filtered_turn_state[domain_slot]['mentioned_idx'] = -1
            if domain_slot_type_map[domain_slot] == 'classify':
                filtered_turn_state[domain_slot]['hit_value'] = -1
            else:
                filtered_turn_state[domain_slot]['hit_value'] = -1, -1
        else:
            filtered_turn_state[domain_slot]['hit_type'] = turn_state[domain_slot]['hit_type']
            filtered_turn_state[domain_slot]['mentioned_idx'] = turn_state[domain_slot]['mentioned_idx']
            filtered_turn_state[domain_slot]['hit_value'] = turn_state[domain_slot]['hit_value']
    return filtered_turn_state


class DataSample(object):
    def __init__(self, sample_id, active_domain, active_slot, filtered_state, label, context, context_mask,
                 mentioned_slot_list):
        self.sample_id = sample_id
        self.active_domain = active_domain
        self.active_slot = active_slot
        self.context = context
        self.context_mask = context_mask
        self.mentioned_slot_list = mentioned_slot_list
        self.label, self.hit_type, self.mentioned_idx, self.hit_value = {}, {}, {}, {}
        for domain_slot in domain_slot_list:
            self.label[domain_slot] = label[domain_slot]
            self.hit_type[domain_slot] = filtered_state[domain_slot]['hit_type']
            self.mentioned_idx[domain_slot] = filtered_state[domain_slot]['mentioned_idx']
            self.hit_value[domain_slot] = filtered_state[domain_slot]['hit_value']


def dialogue_reorganize(dialogue_idx, utterance_list, state_dict, act_dict, classify_slot_value_index_dict):
    reorganize_data = {}
    # dialogue index在此函数中无意义，只是debug时加一个定位参数
    assert len(utterance_list) % 2 == 0 and dialogue_idx is not None
    # 当凭借act中的标签无法确定具体domain时，是不是根据前后文进行辅助判定
    aux_act_assign = args['auxiliary_act_domain_assign']

    history = ''
    history_token = []
    history_token_label = {}
    for domain_slot in domain_slot_list:
        history_token_label[domain_slot] = []

    mentioned_slot_set = set()
    for turn_idx in range(0, len(utterance_list) // 2):
        reorganize_data[turn_idx] = {}
        active_domain, active_slots, inform_info, inform_slot_filled_dict = \
            act_reorganize_and_normalize(act_dict, turn_idx, aux_act_assign)
        reorganize_data[turn_idx]['active_domain'] = active_domain
        reorganize_data[turn_idx]['active_slot'] = active_slots
        if turn_idx == 0:
            assert len(active_domain) == 0 and len(active_slots) == 0

        # 本质上经过之前的工作，我们其实可以判定modified_slots是无关紧要的。因为我们其实每次预测的都是accumulated slot而非单轮改变值
        labels = state_extract(state_dict[turn_idx+1])

        system_utterance = normalize_text('' if turn_idx == 0 else utterance_list[2 * turn_idx - 1].lower())
        system_utterance = delex_text(system_utterance, inform_info) if delex_system_utterance else system_utterance
        user_utterance = normalize_text(utterance_list[2 * turn_idx].lower())
        system_utterance_token, user_utterance_token = tokenize(system_utterance), tokenize(user_utterance)
        current_turn_utterance = user_utterance + ' ' + SEP_token + ' ' + system_utterance
        current_turn_utterance_token = user_utterance_token + [SEP_token] + system_utterance_token
        context_utterance = current_turn_utterance + ' ' + SEP_token + ' ' + history
        context_utterance_token = current_turn_utterance_token + [SEP_token] + history_token
        reorganize_data[turn_idx]['context_utterance'] = CLS_token + ' ' + context_utterance
        reorganize_data[turn_idx]['context_utterance_token'] = [CLS_token] + context_utterance_token

        # mention slot set包含上一轮的mentioned slot与本轮提到的slots
        for domain_slot in domain_slot_list:
            split_idx = domain_slot.find('-')
            domain, slot = domain_slot[: split_idx], domain_slot[split_idx + 1:]
            inform_label = inform_info[domain_slot] if domain_slot in inform_info else 'none'
            if inform_label != 'none':
                mentioned_slot_set.add(str(turn_idx)+'$inform$'+domain+'$'+slot+'$'+inform_label)
        reorganize_data[turn_idx]['mentioned_slot_list'] = list(mentioned_slot_set.copy())

        # label标记的是本轮的cumulative数据的真值
        reorganize_data[turn_idx]['label'] = labels.copy()
        # state标记的是用于模型训练的各种值
        reorganize_data[turn_idx]['state'] = {}
        turn_mentioned_slot_set = set()
        for domain_slot in domain_slot_list:
            reorganize_data[turn_idx]['state'][domain_slot] = {}
            value_label = labels[domain_slot]
            split_idx = domain_slot.find('-')
            domain, slot = domain_slot[: split_idx], domain_slot[split_idx + 1:]

            class_type, mentioned_slot, utterance_token_label, value_index = \
                get_turn_label(value_label, context_utterance_token, domain_slot, mentioned_slot_set,
                               classify_slot_value_index_dict)
            if class_type == 'unpointable':
                # 同样，unpointable也置空history label token, 这个是label本身的错误
                class_type = 'none'

            # 不论我们的模型是否能够成功预测，把所有本轮的label全部置入mentioned slot set
            if value_label != 'none':
                turn_mentioned_slot_set.add(str(turn_idx) + '$label$' + domain + '$' + slot + '$' + value_label)
            reorganize_data[turn_idx]['state'][domain_slot]['class_type'] = class_type
            reorganize_data[turn_idx]['state'][domain_slot]['classify_value'] = value_index
            reorganize_data[turn_idx]['state'][domain_slot]['mentioned_slot'] = mentioned_slot
            reorganize_data[turn_idx]['state'][domain_slot]['context_token_label'] = [0] + utterance_token_label

        # 去重
        mentioned_slot_set = eliminate_replicate_mentioned_slot(mentioned_slot_set.union(turn_mentioned_slot_set))
        history = context_utterance
        history_token = context_utterance_token
    return reorganize_data


def get_turn_label(value_label, context_utterance_token, domain_slot, mentioned_slots, classify_slot_value_index_dict):
    # four types of class info has it's priority
    # 尽可能提供补充标签
    utterance_token_label = [0 for _ in context_utterance_token]

    mentioned_id = 'none'
    value_index = -1
    if value_label == 'none' or value_label == 'dontcare':
        class_type = value_label
    else:
        in_utterance_flag, position, value_index = \
            check_label(value_label, context_utterance_token, domain_slot, classify_slot_value_index_dict)
        is_mentioned, mentioned_id = check_mentioned_slot(value_label, mentioned_slots, domain_slot)

        if in_utterance_flag:
            # if the slot is referred multi times, use the first time it shows in user utterance
            start_idx, end_idx = position[0]
            for i in range(start_idx, end_idx):
                utterance_token_label[i] = 1

        if is_mentioned:
            class_type = 'mentioned'
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
    return class_type, mentioned_id, utterance_token_label, value_index


def eliminate_replicate_mentioned_slot(mentioned_slot_set):
    # 由于我们的设计，两方面可能存在累积，一种是一个utterance中其实没提到某个状态，但是出于继承的原因，我们每次label都会有某个状态出现
    # 另一部分，一个utterance中可能inform的就是真值
    # 对于这种重复，我们的策略是，同样的domain-slot-value配对，只保留最新的一个mention，如果是Inform和Label冲突，仅保留label
    mentioned_slot_dict = {}
    for mentioned_slot in mentioned_slot_set:
        turn_idx, mentioned_type, domain, slot, value = mentioned_slot.strip().split('$')
        key = domain+'$'+slot+'$'+value
        if key not in mentioned_slot_dict:
            mentioned_slot_dict[key] = turn_idx, mentioned_type
        else:
            previous_idx, previous_mentioned_type = mentioned_slot_dict[key]
            if int(turn_idx) > int(previous_idx):
                mentioned_slot_dict[key] = turn_idx, mentioned_type
            elif int(turn_idx) == int(previous_idx) and mentioned_type == 'label' and \
                    previous_mentioned_type == 'inform':
                mentioned_slot_dict[key] = turn_idx, mentioned_type
    new_mention_slot_set = set()
    for key in mentioned_slot_dict:
        turn_idx, mentioned_type = mentioned_slot_dict[key]
        new_mention_slot_set.add(str(turn_idx)+'$'+str(mentioned_type)+'$'+key)
    return new_mention_slot_set


def check_mentioned_slot(value_label, mentioned_slot_set, domain_slot):
    # 此处，我们直接设定只有span case需要用mentioned slot，而其他直接设定为通过数据判定
    # 我们的context length在大多数情况下可以容纳

    valid_list = []
    for mentioned_slot in mentioned_slot_set:
        turn_idx, mentioned_type, domain, slot, value = mentioned_slot.strip().split('$')
        target_slot, slot_item = domain_slot.split('-')[-1], slot.split('-')[-1]
        for item_set in MENTIONED_MAP_LIST:
            if target_slot in item_set and slot_item in item_set:
                if approximate_equal_test(value_label, value, use_variant=variant_flag):
                    valid_list.append((int(turn_idx), mentioned_type, domain, slot, value, mentioned_slot))
    if len(valid_list) == 0:
        return False, 'none'
    elif len(valid_list) == 1:
        return True, valid_list[0][5]
    else:
        valid_list = sorted(valid_list, key=lambda x: x[0])
        if valid_list[-1][0] > valid_list[-2][0]:
            return True, valid_list[-1][5]
        elif valid_list[-1][0] < valid_list[-2][0]:
            raise ValueError('')
        else:
            for index in range(len(valid_list)):
                turn_idx, mentioned_type, domain, slot, value, mentioned_slot = valid_list[len(valid_list)-1-index]
                if domain_slot == domain+'-'+slot:
                    return True, mentioned_slot
            return True, valid_list[-1][5]


def state_extract(state_dict):
    """
    check the semi and the inform slots
    这里的整个逻辑策略是这样，数据集中的state事实上分为两个部分，book和semi
    book和semi中的结构大抵都是 {key: value}的字典，因此可以如下代码进行赋值
    此处有一个特例，book中有一个booked，里面存着的是一个字典再嵌套一个列表。当遍历到booked时，此处的domain_slot会因为不在目标列表中
    而booked_slots的信息会在额外的判定中被优先赋值给state，因此这个代码不会有问题
    """
    domain_slot_value = {domain_slot: 'none' for domain_slot in domain_slot_list}
    for domain in state_dict:
        booked = state_dict[domain]['book']['booked']
        booked_slots = {}
        # Check the booked section
        if len(booked) > 0:  # len of booked larger than 0
            for slot in booked[0]:
                booked_slots[slot] = normalize_label('{}-{}'.format(domain, slot), booked[0][slot])  # normalize labels
        for category in ['book', 'semi']:
            for slot in state_dict[domain][category]:  # s for slot name
                domain_slot = '{}-book-{}'.format(domain, slot) if category == 'book' else '{}-{}'.format(domain, slot)
                domain_slot = domain_slot.lower()
                value_label = normalize_label(domain_slot, state_dict[domain][category][slot])
                # Prefer the slot value as stored in the booked section
                if slot in booked_slots:
                    value_label = booked_slots[slot]
                if domain_slot in domain_slot_list:
                    domain_slot_value[domain_slot] = value_label
    return domain_slot_value


if __name__ == '__main__':
    main()
