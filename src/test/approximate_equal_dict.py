import os
import re
import json
from history_read_data import get_dialogue_info


def main():
    multiwoz_dataset_folder = os.path.abspath('../../resource/multiwoz21')
    label_normalize_path = os.path.join(multiwoz_dataset_folder, 'label_map.json')
    normalize_map = json.load(open(label_normalize_path, 'r'))
    domain_slot_list = normalize_map['slots']
    act_data_path = os.path.join(multiwoz_dataset_folder, 'dialogue_acts.json')
    act_dataset = json.load(open(act_data_path, 'r'))
    approximate_map = normalize_map['label_maps']
    approximate_map = {key: set(approximate_map[key]) for key in approximate_map}

    multiwoz_22 = os.path.abspath('../../resource/multiwoz22/data.json')
    data_dict = json.load(open(multiwoz_22, 'r'))
    raw_data_dict = {}
    for dialogue_idx in data_dict:
        raw_data_dict[dialogue_idx] = get_dialogue_info(act_dataset, data_dict, dialogue_idx)

    for dialogue_idx in raw_data_dict:
        for turn in raw_data_dict[dialogue_idx][1]:  # data_dict[dialogue_idx] = [utterance_list, state, act]
            state_dict = raw_data_dict[dialogue_idx][1][turn]
            for domain in state_dict:
                booked = state_dict[domain]['book']['booked']
                booked_slots = {}
                # Check the booked section
                if len(booked) > 0:  # len of booked larger than 0
                    for slot in booked[0]:
                        booked_slots[slot] = booked[0][slot]
                for category in ['book', 'semi']:
                    for slot in state_dict[domain][category]:  # s for slot name
                        domain_slot = '{}-book-{}'.format(domain, slot) if category == 'book' else '{}-{}'.format(
                            domain, slot)
                        domain_slot = domain_slot.lower()
                        value_label = state_dict[domain][category][slot]
                        # Prefer the slot value as stored in the booked section
                        if slot in booked_slots:
                            value_label = booked_slots[slot]
                        if domain_slot in domain_slot_list:
                            if len(value_label) > 1:
                                for index_1 in range(len(value_label)):
                                    if value_label[index_1] not in approximate_map:
                                        approximate_map[value_label[index_1]] = set()
                                    for index_2 in range(len(value_label)):
                                        if index_2 != index_1:
                                            approximate_map[value_label[index_1]].add(value_label[index_2])
    approximate_map = {key: list(approximate_map[key]) for key in approximate_map}
    json.dump(approximate_map,
              open(os.path.abspath('../../resource/multiwoz22/approximate_test.json'), 'w'))
    json.dump(approximate_map,
              open(os.path.abspath('../../resource/multiwoz21/approximate_test.json'), 'w'))


def state_extract(state_dict, domain_slot_list):
    """
    checked 211206
    提取当前turn的累积state
    check the semi and the inform slots
    这里的整个逻辑策略是这样，数据集中的state事实上分为两个部分，book和semi
    book和semi中的结构大抵都是 {key: value}的字典，因此可以如下代码进行赋值
    此处有一个特例，book中有一个booked，里面存着的是一个字典再嵌套一个列表。当遍历到booked时，此处的domain_slot会因为不在目标列表中被跳过
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


def normalize_label(domain_slot, value_label):
    # checked 211206
    # 根据设计，不同的slot对应不同的标准化方法（比如时间和其他的就不一样），因此要输入具体的slot name
    # Normalization of empty slots
    if isinstance(value_label, str):  # multiwoz 21
        value_label = value_label.strip().lower()
    elif isinstance(value_label, list):  # multiwoz 22
        if len(value_label) == 0 or (not isinstance(value_label[0], str)):
            return 'none'
        value_label = value_label[0].strip().lower()
    else:
        raise ValueError('')

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

if __name__ == '__main__':
    main()
