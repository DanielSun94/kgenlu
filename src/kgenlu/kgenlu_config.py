import argparse
import torch
import os
import logging

# task and model setting
config_name = 'roberta'
# use history 和 no_value_assign_strategy旨在为以下的情况提供判断
# 我们在判定token label 的start index和end index时，其实会出现一种情况，就是token label其实是在历史token里的
# 这个问题在class type不是hit的情况下可能造成一些问题。就是计算loss的时候，是几个分支的loss都会算的
# 当你class type是none时，按照道理来讲预期的span index是空的，因为本身不存在这样的预测目标
# 但是出于历史上存在hit，其实find index可能真的会找到值。如何处理就是一个问题，一种是保留这种情况
# 另一种是归零。归零时，代表start index和end index都是0
# 同样的，referred slot等预测指标也会存在其实不存在的问题，也给定两个情况。一个是把不存在也作为一个预测值（value），另一个是不存在时直接忽略\\
# （label挂-1, miss）
if config_name == 'roberta':
    config = {
        'train_domain': 'hotel$train$restaurant$attraction$taxi',
        'test_domain': 'hotel$train$restaurant$attraction$taxi',
        'pretrained_model': 'roberta',
        'max_length': 512,
        'batch_size': 32,
        'epoch': 30,
        'train_data_fraction': 1.0,
        'encoder_d_model': 768,
        'learning_rate': 0.00001,
        'device': 'cuda:1',
        'auxiliary_domain_assign': True,
        'name': 'kgenlu-roberta',
        'use_history': False,
        'no_value_assign_strategy': 'value',  # value
        'max_grad_norm': 1.0
    }
else:
    raise ValueError('Invalid Config Name')

DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

NONE_IDX, DONTCARE_INDEX, HIT_INDEX = 0, 1, 2

parser = argparse.ArgumentParser(description='Knowledge Graph Enhanced NLU (KGENLU)')
parser.add_argument('--train_domain', help='training domain', default=config['train_domain'], required=False)
parser.add_argument('--train_data_fraction', help='data portion', default=config['train_data_fraction'], required=False)
parser.add_argument('--epoch', help='training epoch', default=config['epoch'], required=False)
parser.add_argument('--learning_rate', help='learning_rate', default=config['learning_rate'], required=False)
parser.add_argument('--encoder_d_model', help='encoder_d_model', default=config['encoder_d_model'], required=False)
parser.add_argument('--batch_size', help='training domain', default=config['batch_size'], required=False)
parser.add_argument('--pretrained_model', help='pretrained_model', default=config['pretrained_model'], required=False)
parser.add_argument('--test_domain', help='test domain', default=config['test_domain'], required=False)
parser.add_argument('--name', help='name', default=config['name'], required=False)
parser.add_argument('--max_grad_norm', help='max_grad_norm', default=config['max_grad_norm'], required=False)
parser.add_argument('--no_value_assign_strategy', help='test domain',
                    default=config['no_value_assign_strategy'], required=False)
parser.add_argument('--use_history', help='use history utterance', default=config['use_history'], required=False)
parser.add_argument('--max_len', help='test domain', default=config['max_length'], required=False)
parser.add_argument('--auxiliary_domain_assign', help='auxiliary_domain_assign',
                    default=config['auxiliary_domain_assign'], required=False)
args = vars(parser.parse_args())


# logger
log_file_name = os.path.abspath('../../resource/log.txt')
FORMAT = "%(asctime)s %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=log_file_name)
console_logger = logging.StreamHandler()
file_logger = logging.FileHandler(log_file_name, mode='a', encoding='UTF-8')
file_logger.setLevel(logging.INFO)
# console output format
stream_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(message)s")
# file output format
logging_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(message)s")
file_logger.setFormatter(logging_format)
console_logger.setFormatter(stream_format)
logger.addHandler(file_logger)
logger.addHandler(console_logger)
logger.info("|------logger.info-----")


# special token
UNK_token, PAD_token, SEP_token, CLS_token = '<unk>', '<pad>', '</s>', '<s>'
DATA_TYPE_UTTERANCE, DATA_TYPE_SLOT, DATA_TYPE_BELIEF = 'utterance', 'slot', 'belief'


# resource path
multiwoz_dataset_folder = os.path.abspath('../../resource/multiwoz')
model_checkpoint_folder = os.path.abspath('../../resource/model_check_point')
# dataset, time, epoch, general acc
result_template = os.path.join(os.path.abspath('../../resource/evaluation'), '{}_{}_epoch_{}_{}.csv')
# train_idx_path = os.path.join(multiwoz_dataset_folder, 'trainListFile.json')
dialogue_data_path = os.path.join(multiwoz_dataset_folder, 'data.json')
dev_idx_path = os.path.join(multiwoz_dataset_folder, 'valListFile.json')
test_idx_path = os.path.join(multiwoz_dataset_folder, 'testListFile.json')
label_normalize_path = os.path.join(multiwoz_dataset_folder, 'label_map.json')
act_data_path = os.path.join(multiwoz_dataset_folder, 'dialogue_acts.json')
dialogue_data_cache_path = os.path.join(multiwoz_dataset_folder, 'dialogue_data_cache_{}.pkl')
# dialogue_unstructured_data_cache_path = os.path.join(multiwoz_dataset_folder, 'dialogue_data_coarse_cache_{}.pkl')
classify_slot_value_index_map_path = os.path.join(multiwoz_dataset_folder, 'classify_slot_value_index_map_path.pkl')


# act
act_type = {
    'inform',
    'request',
    'select',  # for restaurant, hotel, attraction
    'recommend',  # for restaurant, hotel, attraction
    'not found',  # for restaurant, hotel, attraction
    'request booking info',  # for restaurant, hotel, attraction
    'offer booking',  # for restaurant, hotel, attraction, train
    'inform booked',  # for restaurant, hotel, attraction, train
    'decline booking'  # for restaurant, hotel, attraction, train
    # did not use four meaningless act, 'welcome', 'greet', 'bye', 'reqmore'
}
DOMAIN_IDX_DICT = {'restaurant': 0, 'hotel': 1, 'attraction': 2, 'taxi': 3, 'train': 4}
IDX_DOMAIN_DICT = {0: 'restaurant', 1: 'hotel', 2: 'attraction', 3: 'taxi', 4: 'train'}

SLOT_IDX_DICT = {'leaveat': 0, 'destination': 1, 'departure': 2, 'arriveby': 3, 'people': 4, 'day': 5, 'time': 6,
                 'food': 7, 'pricerange': 8, 'name': 9, 'area': 10, 'stay': 11, 'parking': 12, 'stars': 13,
                 'internet': 14, 'type': 15}

IDX_SLOT_DICT = {0: 'leaveat', 1: 'destination', 2: 'departure', 3: 'arriveby', 4: 'people', 5: 'day', 6: 'time',
                 7: 'food', 8: 'pricerange', 9: 'name', 10: 'area', 11: 'stay', 12: 'parking', 13: 'stars',
                 14: 'internet', 15: 'type'}


# Required for mapping slot names in dialogue_acts.json file
# to proper designations.
ACT_SLOT_NAME_MAP_DICT = {'depart': 'departure', 'dest': 'destination', 'leave': 'leaveat', 'arrive': 'arriveby',
                          'price': 'pricerange'}

ACT_MAP_DICT = {
    'taxi-depart': 'taxi-departure',
    'taxi-dest': 'taxi-destination',
    'taxi-leave': 'taxi-leaveat',
    'taxi-arrive': 'taxi-arriveby',
    'train-depart': 'train-departure',
    'train-dest': 'train-destination',
    'train-leave': 'train-leaveat',
    'train-arrive': 'train-arriveby',
    'train-people': 'train-book_people',
    'restaurant-price': 'restaurant-pricerange',
    'restaurant-people': 'restaurant-book-people',
    'restaurant-day': 'restaurant-book-day',
    'restaurant-time': 'restaurant-book-time',
    'hotel-price': 'hotel-pricerange',
    'hotel-people': 'hotel-book-people',
    'hotel-day': 'hotel-book-day',
    'hotel-stay': 'hotel-book-stay',
    'booking-people': 'booking-book-people',
    'booking-day': 'booking-book-day',
    'booking-stay': 'booking-book-stay',
    'booking-time': 'booking-book-time',
}