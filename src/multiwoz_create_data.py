# -*- coding: utf-8 -*-
import json
import os
import re
from typing import List, Any
import difflib
import numpy as np
from multiwoz_config import multiwoz_data_folder, MAX_LENGTH

np.set_printoptions(precision=3)
np.random.seed(2)
IGNORE_KEYS_IN_GOAL = ['eod', 'topic', 'messageLen', 'message']


"""
Sunzhoujian: The preprocess code is mainly from https://github.com/jasonwu0731/trade-dst create_data.py
end comment

Most of the codes are from https://github.com/budzianowski/multiwoz
"""
#
# testListFile = 'testListFileForDebug'
# valListFile = 'valListFileForDebug'
testListFile = 'testListFile.json'
valListFile = 'valListFile.json'

with open(os.path.join(multiwoz_data_folder, 'mapping.pair'), 'r') as fin:
    replacements = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def insertSpace(token, text):
    s_idx = 0
    while True:
        s_idx = text.find(token, s_idx)
        if s_idx == -1:
            break
        if s_idx + 1 < len(text) and re.match('[0-9]', text[s_idx - 1]) and \
                re.match('[0-9]', text[s_idx + 1]):
            s_idx += 1
            continue
        if text[s_idx - 1] != ' ':
            text = text[:s_idx] + ' ' + text[s_idx:]
            s_idx += 1
        if s_idx + len(token) < len(text) and text[s_idx + len(token)] != ' ':
            text = text[:s_idx + 1] + ' ' + text[s_idx + 1:]
        s_idx += 1
    return text


def normalize(text, clean_value=True):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    if clean_value:
        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            s_idx = 0
            for m in ms:
                s_idx = text.find(m[0], s_idx)
                if text[s_idx - 1] == '(':
                    s_idx -= 1
                e_idx = text.find(m[-1], s_idx) + len(m[-1])
                text = text.replace(text[s_idx:e_idx], ''.join(m))

        # normalize postcode
        ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]'
                        '?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', text)
        if ms:
            s_idx = 0
            for m in ms:
                s_idx = text.find(m, s_idx)
                e_idx = s_idx + len(m)
                text = text[:s_idx] + re.sub('[,\. ]', '', m) + text[e_idx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # if clean_value:
    #     # replace time and and price
    #     text = re.sub(time_pat, ' [value_time] ', text)
    #     text = re.sub(price_pat, ' [value_price] ', text)
    #     # text = re.sub(price_pat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for from_x, to_x in replacements:
        text = ' ' + text + ' '
        text = text.replace(from_x, to_x)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def fixDelex(filename, data, data2, idx, idx_acts):
    """
    Given system dialogue acts fix automatic delexicalization.
    data: origin utterance, corresponding belief state, and dialogue goal
    data2: dialogue act
    """
    if data2.__contains__(filename.strip('.json')) and data2[filename.strip('.json')].__contains__(str(idx_acts)):
        turn = data2[filename.strip('.json')][str(idx_acts)]
    else:
        return data

    # and not isinstance(turn, unicode):
    if not isinstance(turn, str):
        for k, act in turn.items():
            if 'Attraction' in k:
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "attraction")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "attraction")
            if 'Hotel' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "hotel")
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "hotel")
            if 'Restaurant' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "restaurant")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "restaurant")

    return data


def get_dialog_act(filename, _, data2, __, idx_acts):
    """
    Given system dialogue acts fix automatic delexicalization
    Only useful in NLP generation problem
    """
    acts = []
    if data2.__contains__(filename.strip('.json')) and data2[filename.strip('.json')].__contains__(str(idx_acts)):
        turn = data2[filename.strip('.json')][str(idx_acts)]
    else:
        return acts

    if not isinstance(turn, str):  # and not isinstance(turn, unicode):
        for k in turn.keys():
            # temp = [k.split('-')[0].lower(), k.split('-')[1].lower()]
            # for a in turn[k]:
            #     acts.append(temp + [a[0].lower()])

            if k.split('-')[1].lower() == 'request':
                for a in turn[k]:
                    acts.append(a[0].lower())
            elif k.split('-')[1].lower() == 'inform':
                for a in turn[k]:
                    acts.append([a[0].lower(), normalize(a[1].lower(), False)])

    return acts


def get_summary_b_state(b_state, get_domain_=False):
    """
    We form structurized multi-domain belief state from unstructured b_state when get_domain_ is false
    or get active domain according to the b_state when get_domain_ is True

    get_domain_ is false
    此处一共存在7个domain，book中总计存在15个slot(包括booked和其他信息)，因此一共存在15次append
    7个domain中存在24个semi值，每个semi值用3-element的list表达，因此存在72个element被插入
    7个domain每个还需要一个ele来表明是否activate，因此存在7个element被插入
    因此一个utterance的state可以用一个94维的二值向量表示，其中每个element都有特定的语义

    book 和 semi的区别，看上去是semi中的slot是可以置为don't care，而book中的slot不可以
    """
    domains = ['taxi', 'restaurant', 'hospital', 'hotel', 'attraction', 'train', 'police']
    summary_b_state = []
    summary_b_value = []
    active_domain = []
    for domain in domains:
        domain_active = False
        booking = []
        # print(domain,len(b_state[domain]['book'].keys()))
        for slot in sorted(b_state[domain]['book'].keys()):
            if slot == 'booked':
                if len(b_state[domain]['book']['booked']) != 0:
                    booking.append(1)
                    # summary_b_value.append("book {} {}:{}".format(domain, slot, "Yes"))
                else:
                    booking.append(0)
            else:
                if b_state[domain]['book'][slot] != "":
                    # (["book", domain, slot, b_state[domain]['book'][slot]])
                    booking.append(1)
                    summary_b_value.append(["{}-book {}".format(domain, slot.strip().lower()),
                                            normalize(b_state[domain]['book'][slot].strip().lower(), False)])
                else:
                    booking.append(0)
        # 按照道理来讲slot应该是完整结构化的。
        # 但是在train 这里稍微有些例外，book中应当存在people和ticket两个element，但是大部分utterance中都存在缺漏
        # 因此如果缺了，就补上，以保证结构化
        if domain == 'train':
            if 'people' not in b_state[domain]['book'].keys():
                booking.append(0)
            elif 'ticket' not in b_state[domain]['book'].keys():
                booking.append(0)
        summary_b_state += booking

        for slot in b_state[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            # not mentioned 也是一种形式的提到，如果真的是完全不在意会置空
            if b_state[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif b_state[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", "do not care"]:
                slot_enc[1] = 1
                # (["semi", domain, slot, "dontcare"])
                summary_b_value.append(["{}-{}".format(domain, slot.strip().lower()), "dontcare"])
            elif b_state[domain]['semi'][slot]:
                # (["semi", domain, slot, b_state[domain]['semi'][slot]])
                # 不知道为什么，所有的相关模型的create代码都漏了这个设2为1的代码段
                slot_enc[2] = 1
                summary_b_value.append(["{}-{}".format(domain, slot.strip().lower()),
                                        normalize(b_state[domain]['semi'][slot].strip().lower(), False)])
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_b_state += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_b_state += [1]
            active_domain.append(domain)
        else:
            summary_b_state += [0]

    # print(len(summary_b_state))
    assert len(summary_b_state) == 94
    if get_domain_:
        return active_domain
    else:
        return summary_b_state, summary_b_value


def analyze_dialogue(dialogue, max_len):
    """
    Cleaning procedure for all kinds of errors in text and annotation.
    注意，Multiwoz是一个偶数句子的对话，是以user, system一次对话为一个turn完成的。
    """
    # do all the necessary postprocessing
    if len(dialogue['log']) % 2 != 0:
        # print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {'goal': dialogue['goal']}
    usr_turns = []
    sys_turns = []
    # last_bvs = []
    for i in range(len(dialogue['log'])):
        if len(dialogue['log'][i]['text'].split()) > max_len:
            print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            text = dialogue['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            usr_turns.append(dialogue['log'][i])
        else:  # sys turn
            text = dialogue['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            belief_summary, belief_value_summary = get_summary_b_state(dialogue['log'][i]['metadata'])
            dialogue['log'][i]['belief_summary'] = str(belief_summary)
            dialogue['log'][i]['belief_value_summary'] = belief_value_summary
            sys_turns.append(dialogue['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    sys_a = [t['dialogue_acts'] for t in d_orig['sys_log']]
    bvs = [t['belief_value_summary'] for t in d_orig['sys_log']]
    domain = [t['domain'] for t in d_orig['usr_log']]
    for item in zip(usr, sys, sys_a, domain, bvs):
        dial.append({'usr': item[0], 'sys': item[1], 'sys_a': item[2], 'domain': item[3], 'bvs': item[4]})
    return dial


def get_domain(idx, log, domains, last_domain):
    """
    Get the active domain of the current turn
    Basic assumption: an arbitrary turn contains information about at least one and at most one domain.
    """
    if idx == 1:
        # As get_domain is invoked in system's turn, the idx of turn is odd number, and the idx of the first
        # system turn is 1
        # metadata recoded the accumulated belief state of dialogue (rather than the state of current turn)
        active_domains = get_summary_b_state(log[idx]["metadata"], True)
        crnt_doms = active_domains[0] if len(active_domains) != 0 else domains[0]
        return crnt_doms
    else:
        ds_diff = get_ds_diff(log[idx-2]["metadata"], log[idx]["metadata"])
        # no clues from dialog states
        if len(ds_diff.keys()) == 0:
            crnt_doms = last_domain
        else:
            crnt_doms: List[Any] = list(ds_diff.keys())
        # print(crnt_doms)
        # How about multiple domains in one sentence scenario ?
        return crnt_doms[0]


def get_ds_diff(prev_d, crnt_d):
    diff = {}
    # Sometimes, metadata is an empty dictionary, bug?
    if not prev_d or not crnt_d:
        return diff

    for ((k1, v1), (k2, v2)) in zip(prev_d.items(), crnt_d.items()):
        assert k1 == k2
        # updated
        if v1 != v2:
            diff[k2] = v2
    return diff


def createData():
    """
    Ensure the raw multiwoz dataset is already in /resource/multiwoz folder
    https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y
    """
    delex_data = {}

    with open(os.path.join(multiwoz_data_folder, 'data.json'), 'r') as fin1:
        data = json.load(fin1)

    with open(os.path.join(multiwoz_data_folder, 'dialogue_acts.json'), 'r') as fin2:
        data2 = json.load(fin2)

    for d_idx, dialogue_name in enumerate(data):
        dialogue = data[dialogue_name]

        # Referred domains in the dialogue
        domains = []
        for domain_key, domain_value in dialogue['goal'].items():
            # check whether contains some goal entities
            if domain_value and domain_key not in IGNORE_KEYS_IN_GOAL:
                domains.append(domain_key)

        idx_acts = 1
        last_domain, last_slot_fill = "", []
        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            origin_text = normalize(turn['text'], False)
            # origin_text = delexicalize.markEntity(origin_text, dic)
            dialogue['log'][idx]['text'] = origin_text

            if idx % 2 == 1:  # if it's a system turn

                current_domain = get_domain(idx, dialogue['log'], domains, last_domain)
                last_domain = [current_domain]

                dialogue['log'][idx - 1]['domain'] = current_domain
                dialogue['log'][idx]['dialogue_acts'] = get_dialog_act(dialogue_name, dialogue, data2, idx, idx_acts)
                idx_acts += 1

            # FIXING delexicalization:
            # fixDelex在这里其实没啥用
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)

        delex_data[dialogue_name] = dialogue
    return delex_data


def buildDelexDict(origin_sent, delex_sent):
    dictionary = {}
    s = difflib.SequenceMatcher(None, delex_sent.split(), origin_sent.split())
    bs = s.get_matching_blocks()
    for i, b in enumerate(bs):
        if i < len(bs)-2:
            a_start = b.a + b.size
            b_start = b.b + b.size
            b_end = bs[i+1].b
            dictionary[a_start] = " ".join(origin_sent.split()[b_start:b_end])
    return dictionary


def divideData(data):
    """
    Given test and validation sets, divide
    the data for three different sets
    注意，原始数据中，每个dialog 都是完整的偶数个utterance，以user sys一问一答形式完成
    这就造成了sys的最后一次问答全部都是无意义的寒暄（bye之类的）。
    在此处的divide中，预处理脚本完成了一次移位，相当于把最后一次寒暄去掉了。
    """
    test_list_file = []
    with open(os.path.join(multiwoz_data_folder, testListFile), 'r') as f_in:
        for line_ in f_in:
            test_list_file.append(line_[:-1])

    val_list_file = []
    with open(os.path.join(multiwoz_data_folder, valListFile), 'r') as f_in:
        for line_ in f_in:
            val_list_file.append(line_[:-1])

    train_list_file = open(os.path.join(multiwoz_data_folder, 'trainListFile.json'), 'w')

    test_dials = []
    val_dials = []
    train_dials = []

    # # dictionaries
    # word_freqs_usr = OrderedDict()
    # word_freqs_sys = OrderedDict()

    count_train, count_val, count_test = 0, 0, 0

    for dialogue_name in data:
        # print dialogue_name
        dial_item = data[dialogue_name]
        domains = []
        for dom_k, dom_v in dial_item['goal'].items():
            # check whether contains some goal entities
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL:
                domains.append(dom_k)

        dial = get_dial(data[dialogue_name])
        if dial:
            # list(set([d['domain'] for d in dial]))
            dialogue = {'dialogue_idx': dialogue_name, 'domains': list(set(domains))}
            last_bs = []
            dialogue['dialogue'] = []

            for turn_i, turn in enumerate(dial):
                # 此处有一个左移位的过程，也就是把 user, system 一个turn改为 system user一个turn。
                # 这么做的原因是我们要理解user的话。因此第一个system turn会置空
                # usr, usr_o, sys, sys_o, sys_a, domain
                turn_dialog = dict()
                turn_dialog['system_transcript'] = dial[turn_i-1]['sys'] if turn_i > 0 else ""
                turn_dialog['turn_idx'] = turn_i
                turn_dialog['belief_state'] = [{"slots": [s], "act": "inform"} for s in turn['bvs']]
                turn_dialog['turn_label'] = [bs["slots"][0] for bs in turn_dialog['belief_state'] if bs not in last_bs]
                turn_dialog['transcript'] = turn['usr']
                turn_dialog['system_acts'] = dial[turn_i-1]['sys_a'] if turn_i > 0 else []
                turn_dialog['domain'] = turn['domain']
                last_bs = turn_dialog['belief_state']
                dialogue['dialogue'].append(turn_dialog)

            if dialogue_name in test_list_file:
                test_dials.append(dialogue)
                count_test += 1
            elif dialogue_name in val_list_file:
                val_dials.append(dialogue)
                count_val += 1
            else:
                train_list_file.write(dialogue_name + '\n')
                train_dials.append(dialogue)
                count_train += 1

    train_list_file.close()
    print("# of dialogues: Train {}, Val {}, Test {}".format(count_train, count_val, count_test))

    # save all dialogues
    with open(os.path.join(multiwoz_data_folder, 'dev_dials.json'), 'w') as f:
        json.dump(val_dials, f, indent=4)

    with open(os.path.join(multiwoz_data_folder, 'test_dials.json'), 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open(os.path.join(multiwoz_data_folder, 'train_dials.json'), 'w') as f:
        json.dump(train_dials, f, indent=4)

    # return word_freqs_usr, word_freqs_sys


def main():
    print('Create WOZ-like dialogues. Get yourself a coffee, this might take a while.')
    delex_data = createData()
    print('Divide dialogues...')
    divideData(delex_data)
    # print('Building dictionaries')
    # buildDictionaries(word_freqs_usr, word_freqs_sys)


if __name__ == "__main__":
    main()
