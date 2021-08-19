import argparse
import torch
import os

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 50
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
# 此处源代码有两个MAX LENGTH设置，不知道是怎么搞的
# MAX_LENGTH = 10
IGNORE_KEYS_IN_GOAL = ['eod', 'topic', 'messageLen', 'message']
multiwoz_data_folder = os.path.abspath('../../dataset/multiwoz')
multiwoz_resource_folder = os.path.abspath('../../resource/multiwoz')
parser = argparse.ArgumentParser(description='TRADE Multi-Domain DST')

# Training Setting
parser.add_argument('-path', '--path', help='path of the file to load', required=False)
parser.add_argument('-sample', '--sample', help='Number of Samples', required=False, default=None)
parser.add_argument('-patience', '--patience', help='', required=False, default=6, type=int)
parser.add_argument('-es', '--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-imbsamp', '--imbalance_sampler', help='', required=False, default=0, type=int)
parser.add_argument('-data_ratio', '--data_ratio', help='', required=False, default=0.02, type=int)
parser.add_argument('-um', '--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-bsz', '--batch_size', help='Batch_size', required=False, type=int, default=2)

# Testing Setting
parser.add_argument('-rundev', '--run_dev_testing', help='', required=False, default=0, type=int)
parser.add_argument('-viz', '--vizualization', help='vizualization', type=int, required=False, default=0)
parser.add_argument('-gs', '--genSample', help='Generate Sample', type=int, required=False, default=0)
parser.add_argument('-evalp', '--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an', '--add_name', help='An add name for the save folder', required=False, default='')
parser.add_argument('-eb', '--eval_batch', help='Evaluation Batch_size', required=False, type=int, default=0)

# Model architecture
parser.add_argument('-gate', '--use_gate', help='', required=False, default=1, type=int)
parser.add_argument('-le', '--load_embedding', help='', required=False, default=True, type=bool)
parser.add_argument('-femb', '--fix_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-paral', '--parallel_decode', help='', required=False, default=0, type=int)

# Model Hyper-Parameters
parser.add_argument('-dec', '--decoder', help='decoder model', required=False, default='kge')
parser.add_argument('-hdd', '--hidden', help='Hidden size', required=False, type=int, default=400)
parser.add_argument('-lr', '--learn', help='Learning Rate', required=False, type=float, default=0.001)
parser.add_argument('-dr', '--drop', help='Drop Out', required=False, type=float, default=0.2)
parser.add_argument('-lm', '--limit', help='Word Limit', required=False, default=-10000)
parser.add_argument('-clip', '--clip', help='gradient clipping', required=False, default=10, type=int)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False,
                    default=0.5)
# parser.add_argument('-l','--layer', help='Layer Number', required=False)

# Unseen Domain Setting
parser.add_argument('-l_ewc', '--lambda_ewc', help='regularization term for EWC loss', type=float, required=False,
                    default=0.01)
parser.add_argument('-fisher_sample', '--fisher_sample', help='number of sample used to approximate fisher mat',
                    type=int, required=False, default=0)
parser.add_argument("--all_model", action="store_true")
parser.add_argument("--domain_as_task", action="store_true")
parser.add_argument('--run_except_4d', help='', required=False, default=1, type=int)
parser.add_argument("--strict_domain", action="store_true")
parser.add_argument('-exceptd', '--except_domain', help='', required=False, default="", type=str)
parser.add_argument('-onlyd', '--only_domain', help='', required=False, default="", type=str)

args = vars(parser.parse_args())
if args["load_embedding"]:
    args["hidden"] = 400
    print("[Warning] Using hidden size = 400 for pretrained word embedding (300 + 100)...")
if args["fix_embedding"]:
    args["add_name"] += "FixEmb"
if args["except_domain"] != "":
    args["add_name"] += "Except" + args["except_domain"]
if args["only_domain"] != "":
    args["add_name"] += "Only" + args["only_domain"]

print(str(args))
