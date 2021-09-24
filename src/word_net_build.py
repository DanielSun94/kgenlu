from nltk.corpus import wordnet as wn
import pickle
import os
import numpy as np
ANTONYM, HYPERNYM, INSTANCE_HYPERNYM, HYPONYM, INSTANCE_HYPONYM, MEMBER_HOLONYM, SUBSTANCE_HOLONYM, PART_HOLONYM, \
    MEMBER_MERONYM, SUBSTANCE_MERONYM, PART_MERONYM, TOPIC_DOMAIN, IN_TOPIC_DOMAIN, REGION_DOMAIN, IN_REGION_DOMAIN, \
    USAGE_DOMAIN, IN_USAGE_DOMAIN, ATTRIBUTE, DERIVATIONALLY_RELATED_FORM, ENTAILMENT, CAUSE, \
    ALSO_SEE, VERB_GROUP, SIMILAR_TOS, PERTAINYM, LEMMA, IS_LEMMA = \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
relation_idx_dict = \
    {'antonyms': ANTONYM, 'hypernyms': HYPERNYM, 'instance_hypernyms': INSTANCE_HYPERNYM, 'hyponyms': HYPONYM,
     'instance_hyponyms': INSTANCE_HYPONYM, 'member_holonyms': MEMBER_HOLONYM, 'substance_holonyms': SUBSTANCE_HOLONYM,
     'part_holonyms': PART_HOLONYM, 'member_meronyms': MEMBER_MERONYM, 'substance_meronyms': SUBSTANCE_MERONYM,
     'part_meronyms': PART_MERONYM, 'topic_domains': TOPIC_DOMAIN, 'in_topic_domains': IN_TOPIC_DOMAIN,
     'region_domains': REGION_DOMAIN, 'in_region_domains': IN_REGION_DOMAIN, 'usage_domains': USAGE_DOMAIN,
     'in_usage_domains': IN_USAGE_DOMAIN, 'attributes': ATTRIBUTE, 'entailments': ENTAILMENT, 'also_sees': ALSO_SEE,
     'causes': CAUSE, 'derivationally_related_forms': DERIVATIONALLY_RELATED_FORM, 'verb_groups': VERB_GROUP,
     'similar_tos': SIMILAR_TOS, 'pertainyms': PERTAINYM, 'lemma': LEMMA, 'is_lemma': IS_LEMMA}
idx_relation_dict = {}
for key in relation_idx_dict:
    idx_relation_dict[relation_idx_dict[key]] = key


def main():
    """
    word之间的semantic relation和word lexical relation 参见
    https://wordnet.princeton.edu/documentation/wninput5wn
    以及 nltk.corpus.reader.wordnet有关_WordNetObject对象及该脚本里的文档
    总的来讲，WordNet中一共存在27种关系（如relation_idx_dict所示）
    2种是同义词集和具体词汇（synset, lemma）之间的关系（LEMMA, IS_LEMMA）
    3种刻画具体词汇之间的关系（lexical relation）分别是：derivationally_related_forms， pertainyms， antonyms,
    也就是词汇派生关系，从属关系，反义词
    剩下的22种是同义词集之间的关系

    每个同义词集都有相应的definition，部分有example（例句）。
    synset, definition, example, lemma, 以及lemma和synset之间的关系（包括synset-synset, syn-lemma, lemma-lemma三种关联），
    就是wordnet中包含的所有信息

    不过在实现上，nltk的文档和wordNet的官方文档，以及其具体实现，存在一定的差异。例如，按照道理来讲lemma之间只有三种关系，但nltk中也提供了
    lemma之间的诸如region_domains这样的函数，并且在小部分情况下能返回实际的连接；nltk的也文档中没有说对于synset实现了region_domains，
    但是代码中也实现了。针对这一问题，在本研究中，我们一律采纳最宽泛的纳入标准：凡是nltk reader代码中能够捕获到的关系，不管是不是在文档中定义了，
    都进行收集
    """
    relation_count = np.zeros(27)
    unique_relation_set = set()
    word_idx_dict = dict()
    idx_word_dict = dict()
    # normalized word只映射到lemma的idx
    normalized_word_idx_dict = dict()
    syn_idx_dict = dict()
    idx_syn_dict = dict()
    syn_definition = dict()
    syn_example_dict = dict()
    unique_normalized_word_set = set()
    lemma_full_name_set = set()
    fact_list = []
    syn_number = 0

    # 遍历所有词汇
    # 阅读代码和文档可以发现，一共有5个合法的pos tag：a, r, n, v, s。分别代表 adjective, adverb, noun, verb, adjective-sat
    # 其中，大部分网上的示例都只用前4个，官方文档中也说只有4个词性的词，不过第五种，ADJ-SAT的确是存在的
    # 其实我没搞太清楚s到底是用来干什么的，但是发现加不加这个TAG对于纳入的词的数量没有任何影响，因此这里就不加了
    pos_tag_list = "a", "r", "n", "v"
    # 文档
    word_idx = 0
    for pos_tag in pos_tag_list:
        syn_set_list = list(wn.all_synsets(pos_tag))
        for syn_set in syn_set_list:
            synset_name = syn_set.name().lower().replace('_', ' ').strip() + '$synset'
            if word_idx_dict.__contains__(synset_name):
                print('synset name duplicate error')
            else:
                word_idx_dict[synset_name] = word_idx
                idx_word_dict[word_idx] = synset_name
                syn_number += 1
                word_idx += 1

            syn_idx_dict[synset_name] = list()
            syn_definition[synset_name] = syn_set.definition()
            syn_example_dict[synset_name] = syn_set.examples()
            lemmas = syn_set.lemmas()
            for lemma in lemmas:
                lemma_full_name = (lemma.synset().name() + '.' + lemma.name()).lower().replace('_', ' ').strip()
                lemma_full_name_set.add(lemma_full_name)
                normalized_name = lemma.name().lower().replace('_', ' ').strip()
                unique_normalized_word_set.add(normalized_name)
                if not word_idx_dict.__contains__(lemma_full_name):
                    word_idx_dict[lemma_full_name] = word_idx
                    idx_word_dict[word_idx] = lemma_full_name
                    use_idx = word_idx
                    word_idx += 1
                else:
                    use_idx = word_idx_dict[lemma_full_name]

                syn_idx_dict[synset_name].append(use_idx)

                if normalized_word_idx_dict.__contains__(normalized_name):
                    normalized_word_idx_dict[normalized_name].add(use_idx)
                else:
                    normalized_word_idx_dict[normalized_name] = set()
                    normalized_word_idx_dict[normalized_name].add(use_idx)
    for synset_name in syn_idx_dict:
        for idx in syn_idx_dict[synset_name]:
            idx_syn_dict[idx] = synset_name

    print('word load accomplish')

    # data structure [idx of head word, relation type, idx of tail word]
    for pos_tag in pos_tag_list:
        syn_set_list = list(wn.all_synsets(pos_tag))
        # 记录synset之间的关联
        for syn_set in syn_set_list:
            head_synset_name = syn_set.name().lower().replace('_', ' ').strip() + '$synset'
            semantic_relation_dict = {
                'hypernyms': syn_set.hypernyms(),
                'instance_hypernyms': syn_set.instance_hypernyms(),
                'hyponyms': syn_set.hyponyms(),
                'instance_hyponyms': syn_set.instance_hyponyms(),
                'member_holonyms': syn_set.member_holonyms(),
                'substance_holonyms': syn_set.substance_holonyms(),
                'part_holonyms': syn_set.part_holonyms(),
                'member_meronyms': syn_set.member_meronyms(),
                'substance_meronyms': syn_set.substance_meronyms(),
                'part_meronyms': syn_set.part_meronyms(),
                'topic_domains': syn_set.topic_domains(),
                'in_topic_domains': syn_set.in_topic_domains(),
                'region_domains': syn_set.region_domains(),
                'in_region_domains': syn_set.in_region_domains(),
                'usage_domains': syn_set.usage_domains(),
                'in_usage_domains': syn_set.in_usage_domains(),
                'attributes': syn_set.attributes(),
                'entailments': syn_set.entailments(),
                'causes': syn_set.causes(),
                'also_sees': syn_set.also_sees(),
                'verb_groups': syn_set.verb_groups(),
                'similar_tos': syn_set.similar_tos(),
            }
            for relation in semantic_relation_dict:
                for tail_synset in semantic_relation_dict[relation]:
                    tail_synset_name = tail_synset.name().lower().replace('_', ' ').strip() + '$synset'
                    unique_code = str(word_idx_dict[head_synset_name]) + '$' + str(relation_idx_dict[relation]) \
                        + '$' + str(word_idx_dict[tail_synset_name])
                    if not unique_relation_set.__contains__(unique_code):
                        fact_list.append([word_idx_dict[head_synset_name], relation_idx_dict[relation],
                                          word_idx_dict[tail_synset_name]])
                        unique_relation_set.add(unique_code)
                        relation_count[relation_idx_dict[relation]] += 1
                    else:
                        print('relation duplicate')

            # 记录lemma的相关relation
            lemmas = syn_set.lemmas()
            for lemma in lemmas:
                # 构建lemma和中心词之间的相关关系
                lemma_full_name = (lemma.synset().name() + '.' + lemma.name()).lower().replace('_', ' ').strip()
                unique_code = str(word_idx_dict[head_synset_name]) + '$' + str(LEMMA) \
                    + '$' + str(word_idx_dict[lemma_full_name])

                # 记录lemma和synset之间的从属关系
                if not unique_relation_set.__contains__(unique_code):
                    fact_list.append([word_idx_dict[head_synset_name], LEMMA, word_idx_dict[lemma_full_name]])
                    fact_list.append([word_idx_dict[lemma_full_name], IS_LEMMA, word_idx_dict[head_synset_name]])
                    unique_relation_set.add(unique_code)
                    unique_relation_set.add(str(word_idx_dict[lemma_full_name]) + '$' + str(IS_LEMMA) + '$' +
                                            str(word_idx_dict[head_synset_name]))
                    relation_count[LEMMA] += 1
                    relation_count[IS_LEMMA] += 1

                # 构建lemma之间的相关关系
                word_relation_dict = {
                    'hypernyms': lemma.hypernyms(),
                    'instance_hypernyms': lemma.instance_hypernyms(),
                    'hyponyms': lemma.hyponyms(),
                    'instance_hyponyms': lemma.instance_hyponyms(),
                    'member_holonyms': lemma.member_holonyms(),
                    'substance_holonyms': lemma.substance_holonyms(),
                    'part_holonyms': lemma.part_holonyms(),
                    'member_meronyms': lemma.member_meronyms(),
                    'substance_meronyms': lemma.substance_meronyms(),
                    'part_meronyms': lemma.part_meronyms(),
                    'topic_domains': lemma.topic_domains(),
                    'in_topic_domains': lemma.in_topic_domains(),
                    'region_domains': lemma.region_domains(),
                    'in_region_domains': lemma.in_region_domains(),
                    'usage_domains': lemma.usage_domains(),
                    'in_usage_domains': lemma.in_usage_domains(),
                    'attributes': lemma.attributes(),
                    'entailments': lemma.entailments(),
                    'causes': lemma.causes(),
                    'also_sees': lemma.also_sees(),
                    'verb_groups': lemma.verb_groups(),
                    'similar_tos': lemma.similar_tos(),
                    'antonyms': lemma.antonyms(),
                    'derivationally_related_forms': lemma.derivationally_related_forms(),
                    'pertainyms': lemma.pertainyms(),
                }

                for relation in word_relation_dict:
                    for lemma_ in word_relation_dict[relation]:
                        tail_name = (lemma_.synset().name() + '.' + lemma_.name()).lower().replace('_', ' ').strip()
                        unique_code = str(word_idx_dict[lemma_full_name]) + '$' + str(relation_idx_dict[relation]) \
                            + '$' + str(word_idx_dict[tail_name])
                        if not unique_relation_set.__contains__(unique_code):
                            fact_list.append([word_idx_dict[lemma_full_name], relation_idx_dict[relation],
                                              word_idx_dict[tail_name]])
                            unique_relation_set.add(unique_code)
                            relation_count[relation_idx_dict[relation]] += 1
                        else:
                            print('relation duplicate')
    print('fact load accomplish')

    save_obj = {
        'relation_idx_dict': relation_idx_dict,
        'idx_relation_dict': idx_relation_dict,
        'relation_count': relation_count,
        'unique_relation_set': unique_relation_set,
        'word_idx_dict': word_idx_dict,
        'idx_word_dict': idx_word_dict,
        'normalized_word_idx_dict': normalized_word_idx_dict,
        'syn_idx_dict': syn_idx_dict,
        'idx_syn_dict': idx_syn_dict,
        'syn_definition': syn_definition,
        'syn_example_dict': syn_example_dict,
        'syn_number': syn_number,
        'fact_list': fact_list,
        'unique_normalized_word_set': unique_normalized_word_set,
        'lemma_full_name_set': lemma_full_name_set
    }
    pickle.dump(save_obj, open(os.path.abspath('../resource/wordnet/wordnet_KG.pkl'), 'wb'))
    _ = pickle.load(open(os.path.abspath('../resource/wordnet/wordnet_KG.pkl'), 'rb'))
    print('accomplish')
    print(np.sum(relation_count))
    for key_ in relation_idx_dict:
        print('key: {}, count: {}'.format(key_, relation_count[relation_idx_dict[key_]]))


if __name__ == '__main__':
    main()
