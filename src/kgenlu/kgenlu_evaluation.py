import torch
import numpy as np
import csv
from datetime import datetime
from kgenlu_config import args, result_template
from kgenlu_read_data import domain_slot_type_map, tokenizer, domain_slot_list


def batch_eval(batch_predict_label_dict, train_batch):
    result = {}
    for domain_slot in domain_slot_list:
        confusion_mat = np.zeros([5, len(batch_predict_label_dict[domain_slot])])  # 4 for tp, tn, fp, fn, pfi
        predict_result = batch_predict_label_dict[domain_slot]
        label_result = [item[domain_slot] if domain_slot in item else 'none' for item in train_batch[3]]
        assert len(label_result) == len(predict_result)
        for idx in range(len(predict_result)):
            predict, label = predict_result[idx], label_result[idx]
            if label != 'none' and predict != 'none' and predict == label:
                confusion_mat[0, idx] = 1
            elif label == 'none' and predict == 'none':
                confusion_mat[1, idx] = 1
            elif label == 'none' and predict != 'none':
                confusion_mat[2, idx] = 1
            elif label != 'none' and predict == 'none':
                confusion_mat[3, idx] = 1
            elif label != 'none' and predict != 'none' and predict != label:
                confusion_mat[4, idx] = 1
            else:
                raise ValueError(' ')
        result[domain_slot] = confusion_mat
    return result


def comprehensive_eval(epoch_result, epoch, data_type):
    data_size = -1
    reorganized_result_dict, slot_result_dict, domain_result_dict = {}, {}, {},
    for domain_slot in domain_slot_list:
        reorganized_result_dict[domain_slot] = []

    for batch_result in epoch_result:
        for domain_slot in batch_result:
            reorganized_result_dict[domain_slot].append(batch_result[domain_slot])
    for domain_slot in domain_slot_list:
        reorganized_result_dict[domain_slot] = np.concatenate(reorganized_result_dict[domain_slot], axis=1)
        data_size = len(reorganized_result_dict[domain_slot][0])

    general_result = np.ones(data_size)
    for domain_slot in domain_slot_list:
        domain_result_dict[domain_slot.strip().split('-')[0]] = np.ones(data_size)

    # data structure of reorganized_result {domain_slot_name: ndarray} ndarray: [sample_size, five prediction type]
    # tp, tn, fp, fn, plfp (positive label false prediction)
    for domain_slot in domain_slot_list:
        slot_tp, slot_tn = reorganized_result_dict[domain_slot][0, :], reorganized_result_dict[domain_slot][1, :]
        slot_correct = np.logical_or(slot_tn, slot_tp)
        general_result *= slot_correct
        domain = domain_slot.strip().split('-')[0]
        domain_result_dict[domain] *= slot_correct

    general_acc = np.sum(general_result) / len(general_result)
    domain_acc_dict = {}
    for domain in domain_result_dict:
        domain_acc_dict[domain] = np.sum(domain_result_dict[domain]) / len(domain_result_dict[domain])

    write_rows = []
    for config_item in args:
        write_rows.append([config_item, args[config_item]])
    result_rows = []
    head = ['category', 'accuracy', 'recall', 'precision', 'tp', 'tn', 'fp', 'fn', 'plfp']
    result_rows.append(head)
    general_acc = str(round(general_acc*100, 2)) + "%"
    result_rows.append(['general', general_acc])
    for domain in domain_acc_dict:
        result_rows.append([domain, str(round(domain_acc_dict[domain]*100, 2))+"%"])
    for domain_slot in domain_slot_list:
        result = reorganized_result_dict[domain_slot]
        tp, tn, fp, fn, plfp = result[0, :], result[1, :], result[2, :], result[3, :], result[4, :]
        recall = str(round(100*np.sum(tp) / (np.sum(tp) + np.sum(fn) + np.sum(plfp)), 2))+"%"
        precision = str(round(100*np.sum(tp) / (np.sum(tp) + np.sum(fp)), 2))+"%"
        accuracy = str(round(100*(np.sum(tp) + np.sum(tn)) / len(tp), 2))+"%"
        tp, tn, fp, fn, plfp = np.sum(tp) / len(tp), np.sum(tn) / len(tn), np.sum(fp) / len(fp), np.sum(fn) / len(fn), \
            np.sum(plfp) / len(plfp)
        tp, tn, fp, fn, plfp = str(round(tp*100, 2))+"%", str(round(tn*100, 2))+"%", str(round(fp*100, 2))+"%",\
            str(round(fn*100, 2))+"%", str(round(plfp*100, 2))+"%"
        result_rows.append([domain_slot, accuracy, recall, precision, tp, tn, fp, fn, plfp])
    for line in result_rows:
        write_rows.append(line)

    now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    with open(result_template.format(data_type, now, epoch, general_acc), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(write_rows)
    return result_rows


def reconstruct_batch_predict_label(domain_slot, predict_hit_type_one_slot, predict_value_one_slot,
                                    predict_referral_one_slot, train_batch, classify_slot_index_value_dict):
    batch_change_label = [item[domain_slot] if domain_slot in item else 'none' for item in train_batch[3]]
    batch_utterance, batch_last_turn_label = train_batch[5], train_batch[10]

    hit_type_predict = torch.argmax(predict_hit_type_one_slot, dim=1).cpu().detach().numpy()
    referral_predict = torch.argmax(predict_referral_one_slot, dim=1).cpu().detach().numpy()

    if domain_slot_type_map[domain_slot] == 'classify':
        hit_value_predict = torch.argmax(predict_value_one_slot, dim=1).cpu().detach().numpy()
    else:
        assert domain_slot_type_map[domain_slot] == 'span'
        start_idx_predict = torch.argmax(predict_value_one_slot[:, :, 0], dim=1).unsqueeze(dim=1)
        end_idx_predict = torch.argmax(predict_value_one_slot[:, :, 1], dim=1).unsqueeze(dim=1)
        hit_value_predict = torch.cat((start_idx_predict, end_idx_predict), dim=1).cpu().detach().numpy()

    reconstructed_label = []
    for item in zip(batch_change_label, batch_utterance, batch_last_turn_label, hit_type_predict, referral_predict,
                    hit_value_predict):
        change_label, utterance, last_turn_label, hit_type_predict_item, referral_predict_item, \
            hit_value_predict_item = item
        if hit_type_predict_item == 0:  # for
            reconstructed_label.append('none')
        elif hit_type_predict_item == 1:
            reconstructed_label.append('dontcare')
        elif hit_type_predict_item == 2:
            reconstructed_label.append(last_turn_label[domain_slot])
        elif hit_type_predict_item == 3:
            if referral_predict_item == 30:
                referral_predict_item = 'none'
            else:
                referral_predict_item = last_turn_label[domain_slot_list[referral_predict_item]]
            reconstructed_label.append(referral_predict_item)
        elif hit_type_predict_item == 4:
            if domain_slot_type_map[domain_slot] == 'classify':
                if hit_value_predict_item == len(classify_slot_index_value_dict[domain_slot]):
                    reconstructed_label.append('none')
                else:
                    reconstructed_label.append(classify_slot_index_value_dict[domain_slot][hit_value_predict_item])
            else:
                assert domain_slot_type_map[domain_slot] == 'span'
                start_idx, end_idx = hit_value_predict_item
                if start_idx <= end_idx:
                    target_utterance = utterance[start_idx: end_idx+1]
                    target_value = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(target_utterance))
                    reconstructed_label.append(target_value)
                else:
                    reconstructed_label.append('none')

        else:
            raise ValueError('invalid value')
    return reconstructed_label
