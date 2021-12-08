import argparse
import torch
import os
import logging

# task and model setting
config_name = 'roberta'
if config_name == 'roberta':
    config = {
        'load_ckpt_path': '',  # os.path.join(os.path.abspath('../../resource/model_checkpoint'), 'no1_9.ckpt'),  #  ''
        'start_epoch': 0,  # = 0
        'process_name': 'no1-history-pure-encoder-test',
        'train_domain': 'hotel$train$restaurant$attraction$taxi',
        'test_domain': 'hotel$train$restaurant$attraction$taxi',
        'pretrained_model': 'roberta-base',
        'max_length': 512,
        'batch_size': 4,
        'epoch': 30,
        'data_fraction': 0.01,
        'encoder_d_model': 768,
        'learning_rate': 0.00001,
        'device': 'cuda:1',
        'auxiliary_act_domain_assign': True,
        'delex_system_utterance': False,
        'use_multi_gpu': True,
        'no_value_assign_strategy': 'value',  # value
        'max_grad_norm': 1.0,
        'gate_weight': 0.5,
        'mentioned_weight': 0.5,
        'span_weight': 0.5,
        'classify_weight': 0.5,
        'overwrite_cache': False,
        'use_label_variant': True,
        'mode': 'train',  # train, eval
        'lock_embedding_parameter': True,
        'mentioned_slot_pool_size': 8
    }
else:
    raise ValueError('Invalid Config Name')


DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='history_selection')
parser.add_argument('--load_ckpt_path', help='', default=config['load_ckpt_path'])
parser.add_argument('--use_label_variant', help='', default=config['use_label_variant'])
parser.add_argument('--mode', help='', default=config['mode'])
parser.add_argument('--lock_embedding_parameter', help='', default=config['lock_embedding_parameter'])
parser.add_argument('--start_epoch', help='', default=config['start_epoch'])
parser.add_argument('--process_name', help='', default=config['process_name'])
parser.add_argument('--overwrite_cache', help='', default=config['overwrite_cache'])
parser.add_argument('--mentioned_slot_pool_size', help='', default=config['mentioned_slot_pool_size'])
parser.add_argument('--train_domain', help='', default=config['train_domain'])
parser.add_argument('--gate_weight', help='', default=config['gate_weight'])
parser.add_argument('--span_weight', help='', default=config['span_weight'])
parser.add_argument('--classify_weight', help='', default=config['classify_weight'])
parser.add_argument('--mentioned_weight', help='', default=config['mentioned_weight'])
parser.add_argument('--delex_system_utterance', help='', default=config['delex_system_utterance'])
parser.add_argument('--multi_gpu', help='', default=config['use_multi_gpu'])
parser.add_argument('--data_fraction', help='', default=config['data_fraction'])
parser.add_argument('--epoch', help='', default=config['epoch'])
parser.add_argument('--learning_rate', help='', default=config['learning_rate'])
parser.add_argument('--encoder_d_model', help='', default=config['encoder_d_model'])
parser.add_argument('--batch_size', help='', default=config['batch_size'])
parser.add_argument('--pretrained_model', help='', default=config['pretrained_model'])
parser.add_argument('--local_rank', default=-1, type=int)  # 多卡时必须要有，由程序自行调用，我们其实不需要管
parser.add_argument('--test_domain', help='', default=config['test_domain'])
parser.add_argument('--max_grad_norm', help='', default=config['max_grad_norm'])
parser.add_argument('--no_value_assign_strategy', help='', default=config['no_value_assign_strategy'])
parser.add_argument('--max_len', help='', default=config['max_length'])
parser.add_argument('--auxiliary_act_domain_assign', help='', default=config['auxiliary_act_domain_assign'])
parser.add_argument("--weight_decay", default=0.0, type=float, help="")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="")
args = vars(parser.parse_args())


DATA_TYPE_UTTERANCE, DATA_TYPE_SLOT, DATA_TYPE_BELIEF = 'utterance', 'slot', 'belief'
UNNORMALIZED_ACTION_SLOT = {'none', 'ref', 'choice', 'addr', 'post', 'ticket', 'fee', 'id', 'phone', 'car', 'open'}
UNK_token, PAD_token, SEP_token, CLS_token = '<unk>', '<pad>', '</s>', '<s>'


# resource path
multiwoz_dataset_folder = os.path.abspath('../../resource/multiwoz')
dialogue_data_path = os.path.join(multiwoz_dataset_folder, 'data.json')
dev_idx_path = os.path.join(multiwoz_dataset_folder, 'valListFile.json')
test_idx_path = os.path.join(multiwoz_dataset_folder, 'testListFile.json')
label_normalize_path = os.path.join(multiwoz_dataset_folder, 'label_map.json')
act_data_path = os.path.join(multiwoz_dataset_folder, 'dialogue_acts.json')


cache_path = os.path.abspath('../../resource/history_selection_cache/dialogue_data_cache.pkl')
model_checkpoint_folder = os.path.abspath('../../resource/model_check_point')
evaluation_folder = os.path.abspath('../../resource/evaluation')
# dataset, time, epoch, general acc
medium_result_template = os.path.join(os.path.abspath('../../resource/evaluation'), '{}_{}_{}_{}.pkl')
ckpt_template = os.path.join(os.path.abspath('../../resource/model_checkpoint'), '{}_{}.ckpt')
result_template = os.path.join(os.path.abspath('../../resource/evaluation'), '{}_{}_epoch_{}_{}.csv')


# logger
log_file_name = os.path.abspath('../../resource/log_{}.txt'.format(config['process_name']))
FORMAT = "%(asctime)s %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=log_file_name)
console_logger = logging.StreamHandler()
# console output format
stream_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")
# file output format
console_logger.setFormatter(stream_format)
logger.addHandler(console_logger)
logger.info("|------logger.info-----")

#
# MENTIONED_MAP_LIST = [
#     {'leaveat', 'arriveby', 'time'},
#     {'destination', 'departure', 'name'},
#     {'people'},
#     {'stay'},  # 指的是呆的时间
#     {'day'},  # 指具体星期几
#     {'food'},
#     {'pricerange'},
#     {'area'},
#     {'parking'},
#     {'stars'},
#     {'internet'},
#     {'type'}
# ]
# mentioned只涉及span，其实classify的准确度可以拉满，没有特别大的必要参考别的
MENTIONED_MAP_LIST_DICT = {
    # source->target
    'taxi-leaveat': {"taxi-leaveat", 'train-arriveby'},
    'taxi-destination': {'taxi-destination', 'restaurant-name', 'attraction-name', 'hotel-name', 'train-departure'},
    'taxi-departure': {'taxi-departure', 'restaurant-name', 'attraction-name', 'hotel-name', 'train-destination'},
    'taxi-arriveby': {'taxi-arriveby', 'train-leaveat'},
    'restaurant-book-people': {'restaurant-book-people'},
    'restaurant-book-day': {'restaurant-book-day'},
    'restaurant-book-time': {'restaurant-book-time', 'taxi-arriveby'},
    'restaurant-food': {'restaurant-food'},
    'restaurant-pricerange': {'restaurant-pricerange'},
    'restaurant-name': {'restaurant-name', 'taxi-destination', 'taxi-departure'},
    'restaurant-area': {'attraction-area'},
    'hotel-book-people': {'hotel-book-people'},
    'hotel-book-day': {'hotel-book-day'},
    'hotel-book-stay': {'hotel-book-stay'},
    'hotel-name': {'hotel-name', 'taxi-destination', 'taxi-departure'},
    'hotel-area': {'hotel-area'},
    'hotel-stars': {'hotel-stars'},
    'hotel-parking': {'hotel-parking'},
    'hotel-pricerange': {'hotel-pricerange'},
    'hotel-type': {'hotel-type'},
    'hotel-internet': {'hotel-internet'},
    'attraction-type': {'attraction-type'},
    'attraction-name': {'attraction-name', 'taxi-destination', 'taxi-departure'},
    'attraction-area': {'attraction-area'},
    'train-book-people': {'train-book-people'},
    'train-arriveby': {'train-arriveby', 'taxi-leaveat'},
    'train-destination': {'train-destination', 'taxi-departure'},
    'train-departure': {'train-departure', 'taxi-destination'},
    'train-leaveat': {'train-leaveat', 'taxi-arriveby'},
    'train-day': {'train-day'}
}

# 以下内容均为直接复制
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
