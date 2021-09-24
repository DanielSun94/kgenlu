import os
import json
import re
import pickle
from multiwoz_create_data import normalize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn


def main():
    multiwoz_file = os.path.abspath('../../resource/multiwoz/data.json')
    word_net_set = read_wordnet()
    utterance_list = read_multiwoz(multiwoz_file)
    coverage_test(utterance_list, word_net_set)


def coverage_test(utterance_list, word_net_set):
    un_match_set = set()
    word_net_dict = dict()
    for key in word_net_set:
        word_net_dict[key] = key.split(' ')

    utterance_list = utterance_list[: len(utterance_list)]
    for utterance_idx, utterance in enumerate(utterance_list):
        if utterance_idx % 5000 == 0:
            print('current utterance index: {}'.format(utterance_idx))
        utterance_split = [item.strip() for item in utterance.split(' ')]
        cover_list = [False for _ in range(len(utterance_split))]
        for key in word_net_set:
            if utterance.find(key) == -1:
                continue
            word = [item.strip() for item in key.split(' ')]
            start_idx_list, end_idx_list = contain(word, utterance_split)
            if len(start_idx_list) != 0 and len(end_idx_list) != 0:
                for i in range(len(start_idx_list)):
                    start_index = start_idx_list[i]
                    end_index = end_idx_list[i]
                    for j in range(start_index, end_index):
                        cover_list[j] = True

        for i in range(len(cover_list)):
            if not cover_list[i]:
                un_match_set.add(utterance_split[i])
    un_match_list = list(un_match_set)
    un_match_list.sort()
    print(len(un_match_set))
    pickle.dump(un_match_set, open(os.path.abspath('../../resource/un_match_set.pkl'), 'wb'))
    return un_match_set


def contain(seed, target):
    start_idx = []
    end_idx = []

    target_index = 0
    while target_index < len(target):
        contain_flag = True
        contain_index = 0
        while contain_index < len(seed):
            if contain_index+target_index >= len(target):
                contain_flag = False
                break
            if target[target_index+contain_index] == seed[contain_index]:
                contain_index += 1
            else:
                contain_flag = False
                break
        if contain_flag:
            start_idx.append(target_index)
            end_idx.append(target_index+len(seed))
        target_index += 1
    return start_idx, end_idx


def read_multiwoz(file_name):
    utterance_list = []
    word_set = set()
    with open(file_name, 'r') as f:
        multiwoz_json = json.load(f)
        for key in multiwoz_json:
            single_dialog_utterance_list = eliminate_unnecessary_info(multiwoz_json[key])
            for utterance in single_dialog_utterance_list:
                utterance_list.append(utterance)
                for item in utterance.strip().split(' '):
                    word_set.add(item)

    print('multiwoz word number: {}'.format(len(word_set)))
    return utterance_list


def eliminate_unnecessary_info(dialog):
    wnl = WordNetLemmatizer()
    utterance_list = []
    log = dialog['log']
    eliminate_set = set()
    for turn in log:
        meta = turn['metadata']
        if len(meta.keys()) == 0:
            continue
        for key in meta:
            if meta[key].__contains__('book') and meta[key]['book'].__contains__('booked') and \
                    len(meta[key]['book']['booked']) != 0:
                book_info = meta[key]['book']['booked'][0]
                if book_info.__contains__('reference') and book_info['reference'] != '':
                    eliminate_set.add(book_info['reference'].lower())
                if book_info.__contains__('name') and book_info['name'] != '':
                    eliminate_set.add(book_info['name'].lower())
                if book_info.__contains__('trainID') and book_info['trainID'] != '':
                    eliminate_set.add(book_info['trainID'].lower())

    for turn in log:
        text = turn['text'].replace('\t', ' ')
        # 删除train id
        text_list = text.split(' ')
        text = ''
        for element in text_list:
            if element.find('TR') == -1:
                text += element + ' '
        text = text.strip()
        text = normalize(text)
        # 删除train id
        # 删除reference code
        # 删除旅馆饭店的名称
        for key in eliminate_set:
            text = text.replace(key, ' ')
        # 删除时间
        ms = re.findall('([0-1]?[0-9]|2[0-3]):([0-5][0-9])', text)
        if ms:
            s_idx = 0
            for m in ms:
                s_idx = text.find(m[0], s_idx)
                if text[s_idx - 1] == '(':
                    s_idx -= 1
                e_idx = text.find(m[-1], s_idx) + len(m[-1])
                text = text.replace(text[s_idx:e_idx], '')
        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            s_idx = 0
            for m in ms:
                s_idx = text.find(m[0], s_idx)
                if text[s_idx - 1] == '(':
                    s_idx -= 1
                e_idx = text.find(m[-1], s_idx) + len(m[-1])
                text = text.replace(text[s_idx:e_idx], '')
        # normalize postcode
        ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]'
                                '?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', text)
        if ms:
            s_idx = 0
            for m in ms:
                s_idx = text.find(m, s_idx)
                e_idx = s_idx + len(m)
                text = text[:s_idx] + text[e_idx:]

        text_list = [wnl.lemmatize(word) for word in text.split(' ')]
        text = ''
        for word in text_list:
            text += word + ' '
        utterance_list.append(text.strip())
    return utterance_list


def read_wordnet():
    pos_tag_list = "a", "s", "r", "n", "v"

    unique_word_set = set()
    for pos_tag in pos_tag_list:
        syn_set_list = list(wn.all_synsets(pos_tag))
        for syn_set in syn_set_list:
            lemmas = syn_set.lemmas()
            for lemma in lemmas:
                antonyms = lemma.antonyms()
                hypernyms = lemma.hypernyms()
                hyponyms = lemma.hyponyms()
                entailments = lemma.entailments()
                part_meronyms = lemma.part_meronyms()
                substance_meronyms = lemma.substance_meronyms()
                member_holonyms = lemma.member_holonyms()
                unique_word_set.add(lemma.name().replace('_', ' ').lower())
                for lemma_ in antonyms:
                    unique_word_set.add(lemma_.name().replace('_', ' ').lower())
                for lemma_ in hypernyms:
                    unique_word_set.add(lemma_.name().replace('_', ' ').lower())
                for lemma_ in hyponyms:
                    unique_word_set.add(lemma_.name().replace('_', ' ').lower())
                for lemma_ in entailments:
                    unique_word_set.add(lemma_.name().replace('_', ' ').lower())
                for lemma_ in part_meronyms:
                    unique_word_set.add(lemma_.name().replace('_', ' ').lower())
                for lemma_ in substance_meronyms:
                    unique_word_set.add(lemma_.name().replace('_', ' ').lower())
                for lemma_ in member_holonyms:
                    unique_word_set.add(lemma_.name().replace('_', ' ').lower())
    print(unique_word_set)
    return unique_word_set


if __name__ == '__main__':
    main()
