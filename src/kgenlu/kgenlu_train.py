import torch
import json
from kgenlu_read_data import prepare_data, Sample, label_normalize_path, DEVICE
from kgenlu_model import KGENLU
from kgenlu_config import args, logger
from torch import nn, optim
from tqdm import tqdm

NORMALIZE_MAP = json.load(open(label_normalize_path, 'r'))
pretrained_model = args['pretrained_model']
name = args['name']
domain_slot_list = NORMALIZE_MAP['slots']
domain_slot_type_map = NORMALIZE_MAP['slots-type']


def train(model, train_loader, dev_loader, classify_slot_value_index_dict, classify_slot_index_value_dict):
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
    lr = args['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(args['epoch']):
        logger.info("Epoch :{}".format(epoch))
        # Run the train function
        model.train()

        full_loss = 0
        batch_count = 0
        for train_batch in tqdm(train_loader):
            batch_count += 1
            optimizer.zero_grad()
            gate_loss, classify_loss, referral_loss, span_loss = 0, 0, 0, 0
            predict_gate, predict_dict, referred_dict = model(train_batch)
            for domain_slot in domain_slot_list:
                predict_hit_type_one_slot = predict_gate[domain_slot]
                predict_value_one_slot = predict_dict[domain_slot]
                predict_referral_one_slot = referred_dict[domain_slot]
                label_hit_type_one_slot = train_batch[7][domain_slot].to(DEVICE)
                label_value_one_slot = train_batch[8][domain_slot].to(DEVICE)
                label_referral_one_slot = train_batch[6][domain_slot].to(DEVICE)

                gate_loss += cross_entropy(predict_hit_type_one_slot, label_hit_type_one_slot)
                referral_loss += cross_entropy(predict_referral_one_slot, label_referral_one_slot)
                if domain_slot_type_map[domain_slot] == 'classify':
                    classify_loss += cross_entropy(predict_value_one_slot, label_value_one_slot)
                else:
                    assert domain_slot_type_map[domain_slot] == 'span'
                    pred_start, pred_end = predict_value_one_slot[:, :, 0], predict_value_one_slot[:, :, 1]
                    label_start, label_end = label_value_one_slot[:, 0], label_value_one_slot[:, 1]
                    span_loss += (cross_entropy(pred_start, label_start) + cross_entropy(pred_end, label_end)) / 2
            loss = gate_loss + classify_loss + referral_loss + span_loss
            full_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
            optimizer.step()

        logger.info('epoch: {}, average_loss: {}'.format(epoch, full_loss / batch_count))

        model.eval()

        # validation and test
        torch.cuda.empty_cache()

        # save model


def main():

    data, classify_slot_value_index_dict, classify_slot_index_value_dict = prepare_data(overwrite=False)
    model = KGENLU(name, pretrained_model, classify_slot_value_index_dict)
    train_loader, dev_loader, _ = data

    train(model, train_loader, dev_loader, classify_slot_value_index_dict, classify_slot_index_value_dict)


if __name__ == '__main__':
    main()
