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
        'process_name': 'no2-history-selection',
        'train_domain': 'hotel$train$restaurant$attraction$taxi',
        'test_domain': 'hotel$train$restaurant$attraction$taxi',
        'pretrained_model': 'roberta-large',
        'max_length': 512,
        'batch_size': 8,
        'epoch': 30,
        'train_data_fraction': 1,
        'encoder_d_model': 1024,
        'learning_rate': 0.000005,
        'device': 'cuda:1',
        'auxiliary_act_domain_assign': True,
        'delex_system_utterance': False,
        'use_multi_gpu': False,
        'no_value_assign_strategy': 'value',  # value
        'max_grad_norm': 1.0,
        'gate_weight': 0.2,
        'span_weight': 0.4,
        'classify_weight': 0.2,
        'referral_weight': 0.2,
        'overwrite_cache': False,
        'use_label_variant': True,
        'mode': 'train'  # train
    }
else:
    raise ValueError('Invalid Config Name')


DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='history_selection')
parser.add_argument('--load_ckpt_path', help='', default=config['load_ckpt_path'])
parser.add_argument('--use_label_variant', help='', default=config['use_label_variant'])
parser.add_argument('--mode', help='', default=config['mode'])
parser.add_argument('--start_epoch', help='', default=config['start_epoch'])
parser.add_argument('--process_name', help='', default=config['process_name'])
parser.add_argument('--overwrite_cache', help='', default=config['overwrite_cache'])
parser.add_argument('--train_domain', help='', default=config['train_domain'])
parser.add_argument('--gate_weight', help='', default=config['gate_weight'])
parser.add_argument('--span_weight', help='', default=config['span_weight'])
parser.add_argument('--classify_weight', help='', default=config['classify_weight'])
parser.add_argument('--referral_weight', help='', default=config['referral_weight'])
parser.add_argument('--delex_system_utterance', help='', default=config['delex_system_utterance'])
parser.add_argument('--multi_gpu', help='', default=config['use_multi_gpu'])
parser.add_argument('--train_data_fraction', help='', default=config['train_data_fraction'])
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
log_file_name = os.path.abspath('../../resource/log_history_selection.txt')
FORMAT = "%(asctime)s %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=log_file_name)
console_logger = logging.StreamHandler()
file_logger = logging.FileHandler(log_file_name, mode='a', encoding='UTF-8')
file_logger.setLevel(logging.INFO)
# console output format
stream_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")
# file output format
logging_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")
file_logger.setFormatter(logging_format)
console_logger.setFormatter(stream_format)
logger.addHandler(file_logger)
logger.addHandler(console_logger)
logger.info("|------logger.info-----")

IDX_SLOT_DICT = {0: 'leaveat', 1: 'destination', 2: 'departure', 3: 'arriveby', 4: 'people', 5: 'day', 6: 'time',
                 7: 'food', 8: 'pricerange', 9: 'name', 10: 'area', 11: 'stay', 12: 'parking', 13: 'stars',
                 14: 'internet', 15: 'type'}

MENTIONED_MAP_LIST = [
    {'leaveat', 'arriveby', 'time'},
    {'destination', 'departure', 'name'},
    {'people'},
    {'stay'},  # 指的是呆的时间
    {'day'},  # 指具体星期几
    {'food'},
    {'pricerange'},
    {'area'},
    {'parking'},
    {'stars'},
    {'internet'},
    {'type'}
]