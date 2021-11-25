import os
import torch
import argparse
import logging


load_cpkt_path = os.path.abspath('../../resource/model_checkpoint/check')
config_name = 'roberta'
if config_name == 'roberta':
    config = {
        # TODO identify start epoch in load_cpkt_name
        'load_cpkt_name': None,  # os.path.join(os.path.abspath('../../resource/model_checkpoint'), 'no1_0.ckpt')
        'process_name': 'no5',
        'train_domain': 'hotel$train$restaurant$attraction$taxi',
        'test_domain': 'hotel$train$restaurant$attraction$taxi',
        'pretrained_model': 'roberta-base',
        'max_length': 512,
        'batch_size': 8,
        'epoch': 30,
        'train_data_fraction': 1,
        'dev_data_fraction': 1,
        'test_data_fraction': 1,
        'encoder_d_model': 1024,
        'learning_rate': 0.00001,
        'device': 'cuda:1',
        'auxiliary_domain_assign': True,
        'use_history_utterance': True,
        'use_history_label': False,
        'delex_system_utterance': False,
        'use_multi_gpu': True,
        'max_grad_norm': 1.0,
        # TODO loss weight, positive sample weight in loss function
        'overwrite_cache': False
    }
else:
    raise ValueError('Invalid Config Name')


parser = argparse.ArgumentParser(description='History Selection DST')
parser.add_argument("--load_cpkt_name", default=config['load_cpkt_name'], required=False, help="")
parser.add_argument("--process_name", default=config['process_name'], required=False, help="")
parser.add_argument("--train_domain", default=config['train_domain'], required=False, help="")
parser.add_argument("--pretrained_model", default=config['pretrained_model'], required=False, help="")
parser.add_argument("--max_length", default=config['max_length'], required=False, help="")
parser.add_argument("--batch_size", default=config['batch_size'], required=False, help="")
parser.add_argument("--epoch", default=config['epoch'], required=False, help="")
parser.add_argument("--train_data_fraction", default=config['train_data_fraction'], required=False, help="")
parser.add_argument("--test_data_fraction", default=config['test_data_fraction'], required=False, help="")
parser.add_argument("--dev_data_fraction", default=config['dev_data_fraction'], required=False, help="")
parser.add_argument("--encoder_d_model", default=config['encoder_d_model'], required=False, help="")
parser.add_argument("--learning_rate", default=config['learning_rate'], required=False, help="")
parser.add_argument("--device", default=config['device'], required=False, help="")
parser.add_argument("--auxiliary_domain_assign", default=config['auxiliary_domain_assign'], required=False, help="")
parser.add_argument("--use_history_utterance", default=config['use_history_utterance'], required=False, help="")
parser.add_argument("--use_history_label", default=config['use_history_label'], required=False, help="")
parser.add_argument("--delex_system_utterance", default=config['delex_system_utterance'], required=False, help="")
parser.add_argument("--use_multi_gpu", default=config['use_multi_gpu'], required=False, help="")
parser.add_argument("--max_grad_norm", default=config['max_grad_norm'], required=False, help="")
parser.add_argument("--overwrite_cache", default=config['overwrite_cache'], required=False, help="")
args = vars(parser.parse_args())


DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")


log_file_name = os.path.abspath('../../resource/log_history_selection.txt')
FORMAT = "%(asctime)s %(message)s"
logger = logging.getLogger()
console_logger = logging.StreamHandler()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=log_file_name)
file_logger = logging.FileHandler(log_file_name, mode='a', encoding='UTF-8')
file_logger.setLevel(logging.INFO)
stream_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")
logging_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")
file_logger.setFormatter(logging_format)
console_logger.setFormatter(stream_format)
logger.addHandler(file_logger)
logger.addHandler(console_logger)
logger.info("|------logger.info-----")

# print args information
args_list = []
for item in args:
    args_list.append('{} value: {}'.format(item, args[item]))
sorted(args_list)
for item in args_list:
    logging.info(item)


# path
multiwoz_dataset_folder = os.path.abspath('../../resource/multiwoz')
preprocessed_cache_path = os.path.join(multiwoz_dataset_folder, 'history_selection_dialogue_data_cache_{}.pkl')
dialogue_data_path = os.path.join(multiwoz_dataset_folder, 'data.json')
dev_idx_path = os.path.join(multiwoz_dataset_folder, 'valListFile.json')
test_idx_path = os.path.join(multiwoz_dataset_folder, 'testListFile.json')
label_normalize_path = os.path.join(multiwoz_dataset_folder, 'label_map.json')
act_data_path = os.path.join(multiwoz_dataset_folder, 'dialogue_acts.json')

# Constant
UNK_token, PAD_token, SEP_token, CLS_token = '<unk>', '<pad>', '</s>', '<s>'
