import argparse
import torch
import os
import logging

# task and model setting
config_name = 'default'
if config_name == 'default':
    config = {
        'train_domain': 'hotel$train$restaurant$attraction$taxi',
        'test_domain': 'hotel$train$restaurant$attraction$taxi',
        'max_length': 512,
        'epoch': 30,
        'device': 'cuda:1',
        'auxiliary_domain_assign': True
    }
else:
    raise ValueError('Invalid Config Name')

DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Knowledge Graph Enhanced NLU (KGENLU)')
parser.add_argument('--train_domain', help='training domain', default=config['train_domain'], required=False)
parser.add_argument('--test_domain', help='test domain', default=config['test_domain'], required=False)
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
UNK_id, PAD_id, SEP_id, CLS_id = 0, 1, 2, 3
UNK_token, PAD_token, SEP_token, CLS_token = '<unk>', '</s>', '<cls>', '<pad>'
DATA_TYPE_UTTERANCE, DATA_TYPE_SLOT, DATA_TYPE_BELIEF = 'utterance', 'slot', 'belief'


# resource path
multiwoz_dataset_folder = os.path.abspath('../../resource/multiwoz')
model_checkpoint_folder = os.path.abspath('../../resource/model_check_point')
# train_idx_path = os.path.join(multiwoz_dataset_folder, 'trainListFile.json')
dialogue_data_path = os.path.join(multiwoz_dataset_folder, 'data.json')
dev_idx_path = os.path.join(multiwoz_dataset_folder, 'valListFile.json')
test_idx_path = os.path.join(multiwoz_dataset_folder, 'testListFile.json')
label_normalize_path = os.path.join(multiwoz_dataset_folder, 'label_map.json')
act_data_path = os.path.join(multiwoz_dataset_folder, 'dialogue_acts.json')
dialogue_data_cache_path = os.path.join(multiwoz_dataset_folder, 'dialogue_data_cache.pkl')


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