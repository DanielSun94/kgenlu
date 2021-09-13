import os

finding_set = {'UNK', "EOS", "SOS", "PAD"}
file_path = os.path.abspath('../../resource/embedding/glove.840B.300d.txt')

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        word = line.split(' ')[0]
        if word in finding_set:
            print(word)
