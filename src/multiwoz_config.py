import argparse
import torch
import os

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

UNK_token, PAD_token, EOS_token, SOS_token = 0, 1, 2, 3
UNK, SOS, EOS, PAD = 'UNK', 'SOS', 'EOS', 'PAD'
DATA_TYPE_UTTERANCE, DATA_TYPE_SLOT, DATA_TYPE_BELIEF = 'utterance', 'slot', 'belief'
# 指代slot可能出现的三种情况，dontcare代表用户无所谓，none代表未提及，span代表有提及，且以句子中matching的方式完成匹配
# classify代表有提及，且以分类形式完成匹配
gate_dict = {"dontcare": 0, "none": 1, "span": 2, 'classify': 3}


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi", 'hospital', 'police']
multiwoz_data_folder = os.path.abspath('../resource/multiwoz')
multiwoz_resource_folder = os.path.abspath('../../resource/multiwoz')
parser = argparse.ArgumentParser(description='Multi-Domain DST')

# Setting
parser.add_argument('-lr', '--learning_rate', help='model learning rate', default=0.01, required=False)
parser.add_argument('-bs', '--batch_size', help='training batch size', default=128, required=False)
parser.add_argument('-trd', '--train_domain', help='training domain',
                    default='hotel$train$restaurant$attraction$taxi$hospital$police', required=False)
parser.add_argument('-ted', '--test_domain', help='testing domain',
                    default='hotel$train$restaurant$attraction$taxi$hospital$police', required=False)
parser.add_argument('-mdf', '--multiwoz_dataset_folder', help='multiwoz dataset folder',
                    default=os.path.abspath('../resource/multiwoz/'), required=False)
parser.add_argument('-imbsamp', '--imbalance_sampler', help='', required=False, default=False, type=bool)
parser.add_argument('-es', '--early_stop', help='early stop', default=True, required=False)
parser.add_argument('-evalp', '--eval_epoch', help='eval epoch index', default=100, type=int, required=False)

args = vars(parser.parse_args())

print(str(args))
