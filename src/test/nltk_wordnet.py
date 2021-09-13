from nltk.corpus import wordnet as wn

ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
pos_tag_list = ADJ, ADJ_SAT, ADV, NOUN, VERB

unique_word_set = set()
for pos_tag in pos_tag_list:
    for syn_set in list(wn.all_synsets(pos_tag)):
        lemmas = syn_set.lemmas()
        unique_word_set.add(syn_set.name())
        for lemma in lemmas:
            unique_word_set.add(lemma.name())
        print(syn_set)

print(unique_word_set)
print(len(unique_word_set))
