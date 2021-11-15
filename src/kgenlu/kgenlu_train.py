import logging
import torch
from kgenlu_read_data import prepare_data, Sample, domain_slot_list, domain_slot_type_map
from kgenlu_model import KGENLU
from kgenlu_config import args, logger, DEVICE
from torch import nn, optim
from tqdm import tqdm
from kgenlu_evaluation import reconstruct_batch_predict_label, batch_eval, comprehensive_eval


def train(model, train_loader, dev_loader, test_loader, classify_slot_index_value_dict):
    lr = args['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(args['epoch']):
        logger.info("Epoch :{}".format(epoch))
        epoch_result = []
        # Run the train function
        model.train()
        full_loss = 0
        batch_count = 0

        for train_batch in tqdm(train_loader):
            batch_count += 1
            optimizer.zero_grad()
            predict_gate, predict_dict, referred_dict = model(train_batch)
            loss, train_batch_predict_label_dict = \
                compute_loss_and_batch_eval(predict_gate, predict_dict, referred_dict, train_batch,
                                            classify_slot_index_value_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
            optimizer.step()
            full_loss += loss.detach().item()
            epoch_result.append(batch_eval(train_batch_predict_label_dict, train_batch))
            del loss, predict_gate, predict_dict, referred_dict  # for possible CUDA out of memory

        logger.info('epoch: {}, average_loss: {}'.format(epoch, full_loss / batch_count))
        result_print(comprehensive_eval(epoch_result, epoch, 'train'))

        # validation and test
        model_eval(model, dev_loader, 'dev', epoch, classify_slot_index_value_dict)
        model_eval(model, test_loader, 'test', epoch, classify_slot_index_value_dict)


def result_print(comprehensive_result):
    for line in comprehensive_result:
        print(line)


def compute_loss_and_batch_eval(predict_gate, predict_dict, referred_dict, train_batch, classify_slot_index_value_dict):
    batch_predict_label_dict = {}
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
    gate_loss, classify_loss, referral_loss, span_loss = 0, 0, 0, 0
    for domain_slot in domain_slot_list:
        predict_hit_type_one_slot = predict_gate[domain_slot]
        predict_value_one_slot = predict_dict[domain_slot]
        predict_referral_one_slot = referred_dict[domain_slot]
        label_hit_type_one_slot = train_batch[7][domain_slot].to(DEVICE)
        label_value_one_slot = train_batch[8][domain_slot].to(DEVICE)
        label_referral_one_slot = train_batch[6][domain_slot].to(DEVICE)

        batch_predict_label_dict[domain_slot] = \
            reconstruct_batch_predict_label(domain_slot, predict_hit_type_one_slot, predict_value_one_slot,
                                            predict_referral_one_slot, train_batch,
                                            classify_slot_index_value_dict)

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
    return loss, batch_predict_label_dict


def model_eval(model, data_loader, data_type, epoch, classify_slot_index_value_dict):
    model.eval()
    result = []
    with torch.no_grad():
        for dev_batch in tqdm(data_loader):
            predict_gate, predict_dict, referred_dict = model(dev_batch)
            _, dev_batch_predict_label_dict = \
                compute_loss_and_batch_eval(predict_gate, predict_dict, referred_dict, dev_batch,
                                            classify_slot_index_value_dict)
            result.append(batch_eval(dev_batch_predict_label_dict, dev_batch))
    result_print(comprehensive_eval(result, epoch, data_type))


def main():
    pretrained_model = args['pretrained_model']
    name = args['name']
    data, classify_slot_value_index_dict, classify_slot_index_value_dict = prepare_data(overwrite=False)
    model = KGENLU(name, pretrained_model, classify_slot_value_index_dict)
    train_loader, dev_loader, test_loader = data

    train(model, train_loader, dev_loader, test_loader, classify_slot_index_value_dict)


if __name__ == '__main__':
    for item in args:
        logging.info('{} value: {}'.format(item, args[item]))
    main()
