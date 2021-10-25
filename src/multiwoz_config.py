import argparse
import torch
import os
import logging

epoch = 40
max_sequence_length = 1000
multi_gpu = True
pretrained_model = None # 'roberta' None
data_portion = 100
d_model = 300
cuda = 'cuda:1'
base_learning_rate = 1 * 1e-4
batch_size = 32
model_checkpoint = ""
local_rank = -1

DEVICE = torch.device(cuda if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
MAX_LENGTH = 50
UNK_token, PAD_token, SEP_token, CLS_token = 0, 1, 2, 3
UNK, SEP, CLS, PAD = '<unk>', '</s>', '<cls>', '<pad>'
DATA_TYPE_UTTERANCE, DATA_TYPE_SLOT, DATA_TYPE_BELIEF = 'utterance', 'slot', 'belief'
# 指代slot可能出现的三种情况，dontcare代表用户无所谓，none代表未提及，span代表有提及，且以句子中matching的方式完成匹配
# classify代表有提及，且以分类形式完成匹配
gate_dict = {"dontcare": 0, "none": 1, "span": 2, 'classify': 3}


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi", 'hospital', 'police']
multiwoz_data_folder = os.path.abspath('../resource/multiwoz')
multiwoz_resource_folder = os.path.abspath('../../resource/multiwoz')
parser = argparse.ArgumentParser(description='Multi-Domain DST')

parser.add_argument('-lor', '--local_rank', help='local_rank', type=int, required=False, default=local_rank)
parser.add_argument('-mg', '--multi_gpu', help='use multiple GPUs', type=bool, required=False, default=multi_gpu)
parser.add_argument('-te', '--train_epoch', help='train_epoch', type=int, required=False, default=epoch)
parser.add_argument('-slr', '--span_loss_ratio', help='span_loss_ratio', type=float, required=False, default=0.8)
parser.add_argument('-wur', '--warm_up_ratio', help='warm_up_ratio', type=float, required=False, default=0.1)
parser.add_argument('-msl', '--max_sentence_length', help='max_sentence_length', type=int, required=False,
                    default=max_sequence_length)
parser.add_argument('-tfd', '--training_data_fraction', help='training_data_fraction', type=int, required=False,
                    default=data_portion)
# parser.add_argument('-clip', '--clip', help='gradient clip', default=10, required=False)
parser.add_argument('-sl', '--span_limit', help='classify slot / span slot threshold', default=10, required=False)
parser.add_argument('-esp', '--evaluation_save_folder', help='evaluation save folder', type=str, required=False,
                    default=os.path.abspath('../resource/evaluation/'))
parser.add_argument('-trdf', '--train_data_fraction', help='train data fraction', type=float, required=False,
                    default=1)
parser.add_argument('-tedf', '--test_data_fraction', help='test_data_fraction', type=float, required=False,
                    default=1)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--pretrained_model", default=pretrained_model, type=str, help="pretrained_model")
parser.add_argument("--model_checkpoint_folder", type=str, help="checkpoint folder",
                    default=os.path.abspath('../resource/model_check_point'))
parser.add_argument("--model_checkpoint_name", type=str, help="checkpoint file", default=model_checkpoint)
# Setting
parser.add_argument('-lr', '--learning_rate', help='model learning rate', default=base_learning_rate, required=False,
                    type=float)
parser.add_argument('-bs', '--batch_size', help='training batch size', default=batch_size, required=False)
# hotel$train$restaurant$attraction$taxi$hospital$police
parser.add_argument('-trd', '--train_domain', help='training domain',
                    default='hotel$train$restaurant$attraction$taxi', required=False)
parser.add_argument('-ted', '--test_domain', help='testing domain',
                    default='hotel$train$restaurant$attraction$taxi', required=False)
parser.add_argument('-mdf', '--multiwoz_dataset_folder', help='multiwoz dataset folder',
                    default=os.path.abspath('../resource/multiwoz/'), required=False)
parser.add_argument('-es', '--early_stop', help='early stop', default=True, required=False)
parser.add_argument('-cp', '--cache_path', help='corpus cache path', type=str, required=False,
                    default=os.path.abspath('../resource/multiwoz/cache.pkl'))

# Pretrained Embedding Setting
parser.add_argument('-le', '--load_embedding', help='load pretrained embedding', default=True, type=bool,
                    required=False)
parser.add_argument('-aep', '--aligned_embedding_path', help='folder path of aligned_pretrained embedding',
                    required=False, type=str, default=os.path.abspath('../resource/multiwoz/'))
parser.add_argument('-ue', '--update_embedding', help='update embedding in training phase', default=False, type=bool,
                    required=False)
parser.add_argument('-fep', '--full_embedding_path', help='file path of full embedding', type=str, required=False,
                    default=os.path.abspath('../resource/embedding/glove.42B.300d.txt'))
parser.add_argument('-wnep', '--wordnet_embedding_path', help='pretrained wordnet file path', type=str, required=False,
                    default=os.path.abspath('../resource/transe_checkpoint/WordNet_2000_checkpoint.tar'))
parser.add_argument('-wnp', '--wordnet_path', help='wordnet file path', type=str, required=False,
                    default=os.path.abspath('../resource/wordnet/wordnet_KG.pkl'))

# Encoder (Transformer) Setting
parser.add_argument('-ed', '--encoder_dropout', help='encoder hidden size', default=0.1, type=float, required=False)
parser.add_argument('-edm', '--encoder_d_model', default=d_model, type=int, required=False,
                    help='the number of expected features in the input (token embedding dimension)')
parser.add_argument('-enh', '--encoder_n_head', help='number of multi-head attention', default=6, type=int,
                    required=False)
parser.add_argument('-ea', '--encoder_activation', help='encoder activation function',
                    default='relu', type=str, required=False)
parser.add_argument('-enel', '--encoder_num_encoder_layers', help='number of transformer block',
                    default=6, type=int, required=False)
parser.add_argument('-edff', '--encoder_dim_feed_forward', help='encoder dimension of feed forward layer',
                    default=1024, type=int, required=False)
parser.add_argument('-ehs', '--encoder_hidden_size', help='encoder hidden size',
                    default=128, type=int, required=False)
args = vars(parser.parse_args())

# logger
log_file_name = os.path.abspath('../resource/test_log.log')
FORMAT = "%(asctime)s %(message)s"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO,format=FORMAT, filename=log_file_name)
console_logger = logging.StreamHandler()  # 日志控制台输出

file_logger = logging.FileHandler(log_file_name, mode='a', encoding='UTF-8')
file_logger.setLevel(logging.INFO)

# 控制台输出格式
stream_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")

# 文件输出格式
logging_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")

file_logger.setFormatter(logging_format)
console_logger.setFormatter(stream_format)

logger.addHandler(file_logger)
logger.addHandler(console_logger)

logger.info("|------logger.info-----")

for key in args:
    logging.info('{}: {}'.format(key, args[key]))
