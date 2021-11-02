import os
import pickle
import json
import re
from kgenlu_config import args, logger, dev_idx_path, test_idx_path, act_data_path, ACT_SLOT_NAME_MAP_DICT, \
    label_normalize_path, dialogue_data_cache_path, dialogue_data_path, SEP_token, CLS_token, DOMAIN_IDX_DICT, \
    SLOT_IDX_DICT, ACT_MAP_DICT, UNK_token


NORMALIZE_MAP = json.load(open(label_normalize_path, 'r'))
slot_list = NORMALIZE_MAP['slots']
train_domain = args['train_domain']
test_domain = args['test_domain']
train_domain_set = set(train_domain.strip().split('$'))
test_domain_set = set(test_domain.strip().split('$'))
active_slot_count = dict()


def prepare_data(read_from_cache=False):
    if read_from_cache:
        if os.path.exists(dialogue_data_cache_path):
            return pickle.load(open(dialogue_data_cache_path, 'rb'))
        else:
            raise FileNotFoundError()
    train_idx_list, dev_idx_list, test_idx_list = get_dataset_idx(dev_idx_path, test_idx_path, dialogue_data_path)
    idx_dict = {'train': train_idx_list, 'dev': dev_idx_list, 'test': test_idx_list}

    dialogue_data = json.load(open(dialogue_data_path, 'r'))
    act_data = json.load(open(act_data_path, 'r'))

    train_slots, test_slots = get_slot_list(NORMALIZE_MAP['slots'], train_domain_set, test_domain_set)
    dataset = {
        'train': process_data(idx_dict['train'], dialogue_data, act_data, train_slots),
        'dev': process_data(idx_dict['dev'], dialogue_data, act_data, test_slots),
        'test': process_data(idx_dict['test'], dialogue_data, act_data, test_slots)
    }
    return dataset


def process_data(idx_list, dialogue_dict, act, interest_slots):
    data_dict = []
    idx_set = set(idx_list)
    for dialogue_idx in dialogue_dict:
        if dialogue_idx not in idx_set:
            continue
        if dialogue_idx.strip().split('.')[0] not in act:
            logger.info('act of {} not found'.format(dialogue_idx))

        utterance_list, state_dict, act_dict = get_dialogue_info(act, dialogue_dict, dialogue_idx)
        reorganized_data = dialogue_reorganize_and_normalize(utterance_list, state_dict, act_dict)
    print(active_slot_count)
    return data_dict


def normalize_data(utterance_list, state_dict, act_dict):
    for i in range(len(utterance_list)):
        utterance_list[i] = normalize_text(utterance_list[i])

    return utterance_list, state_dict, act_dict


def dialogue_reorganize_and_normalize(utterance_list, state_dict, act_dict):
    reorganize_data = {}
    assert len(utterance_list) % 2 == 0
    cumulative_labels = {slot: 'none' for slot in slot_list}
    history = ''
    for turn_idx in range(0, len(utterance_list)//2):
        reorganize_data[turn_idx] = {}

        active_domain, active_slots, inform_info = act_reorganize(act_dict, turn_idx+1, args['auxiliary_domain_assign'])
        modified_slots, cumulative_labels = turn_label_reorganize(cumulative_labels, state_dict[turn_idx+1])
        # 按照设计，先整理act, 判断active domain和部分信息，为之后的delex确定
        # The last system utterance is discard for the left shift
        # reorganize and normalize the utterance
        system_utterance = normalize_text('' if turn_idx == 0 else utterance_list[2*turn_idx-1].lower())
        system_utterance = delex_text(system_utterance, inform_info)
        user_utterance = normalize_text(utterance_list[turn_idx].lower())
        current_turn_utterance = system_utterance + ' ' + SEP_token + ' ' + user_utterance
        reorganize_data[turn_idx]['current_turn_utterance'] = CLS_token + ' ' + current_turn_utterance + ' ' + SEP_token
        reorganize_data[turn_idx]['history_utterance'] = CLS_token + ' ' + history + ' ' + SEP_token
        history += current_turn_utterance + ' ' + SEP_token

    return reorganize_data


def delex_text(utterance, values, unk_token=UNK_token):
    utt_norm = tokenize(utterance)
    for slot, value in values.items():
        if value is not 'none':
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


def turn_label_reorganize(cumulative_labels, state_dict):
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
                if s in slot_list and cumulative_labels[s] != value_label:
                    modified_slots[s] = value_label
                    cumulative_labels[s] = value_label
    return modified_slots.copy(), cumulative_labels.copy()


def act_reorganize(act_dict, turn_idx, auxiliary_domain_assign):
    active_domain, active_slots, inform_info = set(), set(), dict()
    if not isinstance(act_dict[turn_idx], dict):
        #  the act dict is string "no annotation" in some cases
        return active_domain, active_slots, inform_info

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
            if act_domain == 'booking': # did not consider booking case
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
                        inform_info[domain_slot] = value
    return active_domain, active_slots, inform_info


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


def utterance_normalize():
    pass


def label_normalize():
    pass


def filter_domain(state_dict, act_dict, interest_slots):
    pass


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


def load_target_dialogue_and_act(dialogue_path, act_path):
    raise NotImplementedError('')


class DSTSample(object):
    def __init__(self, sample_id):
        super(DSTSample)
        self.id = sample_id


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
    if "leaveAt" in slot or "arriveBy" in slot or slot == 'restaurant-book_time':
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub("guesthouse", "guest house", value_label)

    return value_label


def main():
    read_from_cache = False
    logger.info('label_map load success')
    prepare_data(read_from_cache=read_from_cache)
    logger.info('data read success')


if __name__ == '__main__':
    main()