import torch
import numpy as np
import csv
from history_config import args, result_template, MENTIONED_MAP_LIST
from history_read_data import domain_slot_type_map, tokenizer, domain_slot_list, approximate_equal_test, \
    eliminate_replicate_mentioned_slot, get_str_id

use_variant = args['use_label_variant']
mentioned_slot_pool_size = args['mentioned_slot_pool_size']
id_cache_dict = {}


def batch_eval(batch_predict_label_dict, batch):
    result = {}
    for domain_slot in domain_slot_list:
        confusion_mat = np.zeros([5, len(batch_predict_label_dict[domain_slot])])  # 4 for tp, tn, fp, fn, pfi
        predict_result = batch_predict_label_dict[domain_slot]
        # 注意，此处应该使用cumulative label
        label_result = [item for item in batch[5][domain_slot]]
        assert len(label_result) == len(predict_result)
        for idx in range(len(predict_result)):
            predict, label = predict_result[idx], label_result[idx]
            equal = approximate_equal_test(predict, label, use_variant)
            if label != 'none' and predict != 'none' and equal:
                confusion_mat[0, idx] = 1
            elif label == 'none' and predict == 'none':
                confusion_mat[1, idx] = 1
            elif label == 'none' and predict != 'none':
                confusion_mat[2, idx] = 1
            elif label != 'none' and predict == 'none':
                confusion_mat[3, idx] = 1
            elif label != 'none' and predict != 'none' and not equal:
                confusion_mat[4, idx] = 1
            else:
                raise ValueError(' ')
        result[domain_slot] = confusion_mat
    return result


def comprehensive_eval(result_list, data_type, process_name, epoch):
    data_size = -1
    reorganized_result_dict, slot_result_dict, domain_result_dict = {}, {}, {},
    for domain_slot in domain_slot_list:
        reorganized_result_dict[domain_slot] = []

    for batch_result in result_list:
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
    general_acc = str(round(general_acc * 100, 2)) + "%"
    result_rows.append(['general', general_acc])
    for domain in domain_acc_dict:
        result_rows.append([domain, str(round(domain_acc_dict[domain] * 100, 2)) + "%"])
    for domain_slot in domain_slot_list:
        result = reorganized_result_dict[domain_slot]
        tp, tn, fp, fn, plfp = result[0, :], result[1, :], result[2, :], result[3, :], result[4, :]
        recall = str(round(100 * np.sum(tp) / (np.sum(tp) + np.sum(fn) + np.sum(plfp)), 2)) + "%"
        precision = str(round(100 * np.sum(tp) / (np.sum(tp) + np.sum(fp)), 2)) + "%"
        accuracy = str(round(100 * (np.sum(tp) + np.sum(tn)) / len(tp), 2)) + "%"
        tp, tn, fp, fn, plfp = np.sum(tp) / len(tp), np.sum(tn) / len(tn), np.sum(fp) / len(fp), np.sum(fn) / len(fn), \
                               np.sum(plfp) / len(plfp)
        tp, tn, fp, fn, plfp = str(round(tp * 100, 2)) + "%", str(round(tn * 100, 2)) + "%", str(
            round(fp * 100, 2)) + "%", \
                               str(round(fn * 100, 2)) + "%", str(round(plfp * 100, 2)) + "%"
        result_rows.append([domain_slot, accuracy, recall, precision, tp, tn, fp, fn, plfp])
    for line in result_rows:
        write_rows.append(line)

    with open(result_template.format(data_type, process_name, epoch,
                                     general_acc), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(write_rows)
    return result_rows


def reconstruct_prediction_train(domain_slot, predict_hit_type, predict_value, predict_mentioned_slot, data,
                                 slot_index_value_dict):
    # 此处label reconstruct中，train和test的主要区别在于。train中我们默认有上一个turn的label真值，因此在命中referral时，我们
    # 可以直接基于真值进行选择。也因此，我们在train data中采取的是乱序读取策略。而在test中，我们显然不会有上一个label的真值
    # 所以，mentioned slot的命中策略比较麻烦。我们要按照顺序读取数据，然后用上一轮的state来为下一轮的结果提供参照。
    batch_utterance, batch_mentioned_slots = data[3], data[9][domain_slot]

    batch_predict_hit_type = torch.argmax(predict_hit_type, dim=1).cpu().detach().numpy()
    batch_predict_mentioned_slot = torch.argmax(predict_mentioned_slot, dim=1).cpu().detach().numpy()

    if domain_slot_type_map[domain_slot] == 'classify':
        batch_hit_value_predict = torch.argmax(predict_value, dim=1).cpu().detach().numpy()
    else:
        assert domain_slot_type_map[domain_slot] == 'span'
        start_idx_predict = torch.argmax(predict_value[:, :, 0], dim=1).unsqueeze(dim=1)
        end_idx_predict = torch.argmax(predict_value[:, :, 1], dim=1).unsqueeze(dim=1)
        batch_hit_value_predict = torch.cat((start_idx_predict, end_idx_predict), dim=1).cpu().detach().numpy()

    reconstructed_label_list = []
    for item in zip(batch_utterance, batch_mentioned_slots, batch_predict_hit_type, batch_predict_mentioned_slot,
                    batch_hit_value_predict):
        utterance, mentioned_slots, predict_hit_type, predict_mentioned_slot, hit_value_predict = item
        reconstructed_label = predict_label_reconstruct(
            utterance, mentioned_slots, predict_hit_type, predict_mentioned_slot, hit_value_predict, domain_slot,
            slot_index_value_dict)
        reconstructed_label_list.append(reconstructed_label)
    return reconstructed_label_list


def evaluation_test_eval(predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict, batch,
                         slot_index_value_dict, last_mentioned_slot_dict, last_sample_id,
                         last_mentioned_mask_dict, last_str_mentioned_slot_dict):
    batch_predict_label_dict = {}
    for domain_slot in domain_slot_list:
        batch_predict_label_dict[domain_slot] = []
        predict_gate_dict[domain_slot] = predict_gate_dict[domain_slot].cpu().detach().numpy()
        predict_value_dict[domain_slot] = predict_value_dict[domain_slot].cpu().detach().numpy()
        predict_mentioned_slot_dict[domain_slot] = predict_mentioned_slot_dict[domain_slot].cpu().detach().numpy()

    # 由于规定了batch 长度为1，因此此处batch中无需做循环遍历，直接index取0即可
    current_turn_index = str(int(batch[0][0].lower().split('.json-')[1].strip()))
    for domain_slot in domain_slot_list:
        utterance = batch[3][0]
        predict_hit_type_one_slot = predict_gate_dict[domain_slot][0]
        predict_value_one_slot = predict_value_dict[domain_slot][0]
        predict_mentioned_slot = predict_mentioned_slot_dict[domain_slot][0]
        hit_type_predict = int(np.argmax(predict_hit_type_one_slot))
        last_mentioned_slot = last_mentioned_slot_dict[domain_slot]
        predicted_mentioned_slot_idx = int(np.argmax(predict_mentioned_slot))

        if domain_slot_type_map[domain_slot] == 'classify':
            hit_value_predict = int(np.argmax(predict_value_one_slot))
        else:
            assert domain_slot_type_map[domain_slot] == 'span'
            start_idx_predict = int(np.argmax(predict_value_one_slot[:, 0]))
            end_idx_predict = int(np.argmax(predict_value_one_slot[:, 1]))
            hit_value_predict = [start_idx_predict, end_idx_predict]

        predicted_value = predict_label_reconstruct(utterance, last_mentioned_slot, hit_type_predict,
                                                    predicted_mentioned_slot_idx, hit_value_predict, domain_slot,
                                                    slot_index_value_dict)
        batch_predict_label_dict[domain_slot].append(predicted_value)
    last_mentioned_slot_dict, last_mentioned_mask_dict, last_str_mentioned_slot_dict = mentioned_slot_update(
        current_turn_index, batch_predict_label_dict, last_mentioned_slot_dict, last_mentioned_mask_dict,
        last_str_mentioned_slot_dict, 'label')
    return batch_predict_label_dict, last_sample_id, last_mentioned_slot_dict, last_mentioned_mask_dict, \
        last_str_mentioned_slot_dict


def mentioned_slot_update(current_turn_index, update_label_dict, last_mentioned_slot_dict, last_mentioned_mask_dict,
                          last_str_mentioned_slot_dict, mentioned_type):
    # 注意，none, dontcare和<pad>哪怕预测到了，我们也不做mentioned slot看
    # 这一设定与预处理时一致，如果是inform要求domain_slot完全一致，如果是label仅要求在一个区间即可
    skip_value = {'<pad>', 'dontcare', 'none'}
    candidate_mentioned_slot_list_dict = {domain_slot: set() for domain_slot in domain_slot_list}

    # 根据Update值设定candidate
    for domain_slot in domain_slot_list:
        predicted_value = update_label_dict[domain_slot][0]
        source_domain, source_slot = domain_slot.split('-')[0], domain_slot.split('-')[-1]
        if predicted_value not in skip_value:
            for domain_slot_target in domain_slot_list:
                target_domain, target_slot = domain_slot_target.split('-')[0], domain_slot_target.split('-')[-1]
                add_flag = False
                if mentioned_type == 'label':
                    for item in MENTIONED_MAP_LIST:
                        if target_slot in item and source_slot in item:
                            add_flag = True
                elif mentioned_type == 'inform':
                    if target_slot == source_slot and target_domain == target_domain:
                        add_flag = True
                else:
                    raise ValueError('')
                if add_flag:
                    candidate_str = str(
                        current_turn_index) + '$' + mentioned_type + '$' + source_domain + '$' + source_slot + '$' + \
                                    predicted_value
                    candidate_mentioned_slot_list_dict[domain_slot_target].add(candidate_str)
    # 然后按照降序填入最新的previous mentioned value
    for domain_slot in domain_slot_list:
        last_str_mentioned_slot = last_str_mentioned_slot_dict[domain_slot]
        last_mentioned_mask = last_mentioned_mask_dict[domain_slot]
        last_mentioned_slot = last_mentioned_slot_dict[domain_slot]
        for (value_id, mask, str_value) in zip(last_mentioned_slot, last_mentioned_mask, last_str_mentioned_slot):
            if str_value[0] != '<pad>':  # 如果id不为pad(即该slot是真实存在的值而非填充值)，则填入
                assert mask
                idx, mention_type, domain, slot, value = str_value
                if mention_type == 'label':
                    # 如果current index为0，按照道理来讲应该不会进入这步
                    assert int(idx) < int(current_turn_index)
                else:
                    assert int(idx) <= int(current_turn_index)
                candidate_str = str(idx) + '$' + mention_type + '$' + domain + '$' + slot + '$' + value
                candidate_mentioned_slot_list_dict[domain_slot].add(candidate_str)
    # 去重
    for domain_slot in domain_slot_list:
        candidate_mentioned_slot_list_dict[domain_slot] = eliminate_replicate_mentioned_slot(
            candidate_mentioned_slot_list_dict[domain_slot])
        candidate_mentioned_slot_list_dict[domain_slot] = \
            [item.strip().split('$') for item in candidate_mentioned_slot_list_dict[domain_slot]]

    # 根据update值重设三个指标
    updated_mentioned_slot_dict, updated_mentioned_mask_dict, updated_str_mentioned_slot_dict = {}, {}, {}
    for domain_slot in domain_slot_list:
        updated_mentioned_slot_dict[domain_slot] = \
            [[[1], [1], [1], [1], [1]] for _ in range(mentioned_slot_pool_size)]
        updated_str_mentioned_slot_dict[domain_slot] = \
            [['<pad>', '<pad>', '<pad>', '<pad>', '<pad>'] for _ in range(mentioned_slot_pool_size)]
    # 初始化后，先填入本轮更新的值，然后依次填入之前mentioned的值，如果超出最大容限，则取turn index最大的
    # length dict的值从1取起，默认第一个为None
    length_dict = {domain_slot: 1 for domain_slot in domain_slot_list}
    for domain_slot in domain_slot_list:
        valid_list = sorted(candidate_mentioned_slot_list_dict[domain_slot], key=lambda x: int(x[0]))
        # 如果超限，则取最新的
        valid_list = valid_list if len(valid_list) <= mentioned_slot_pool_size - length_dict[domain_slot] else \
            valid_list[-(mentioned_slot_pool_size - length_dict[domain_slot]):]
        for item in valid_list:
            turn, mentioned_type, domain, slot, value = item
            turn_id = get_str_id(turn)
            domain_id = get_str_id(domain)
            slot_id = get_str_id(slot)
            value_id = get_str_id(value)
            mentioned_id = get_str_id(mentioned_type)
            # 根据有效值进行替换
            updated_mentioned_slot_dict[domain_slot][length_dict[domain_slot]] = \
                [turn_id, mentioned_id, domain_id, slot_id, value_id]
            updated_str_mentioned_slot_dict[domain_slot][length_dict[domain_slot]] = \
                [turn, mentioned_type, domain, slot, value]
            length_dict[domain_slot] += 1
        # mask赋值
        updated_mentioned_mask_dict[domain_slot] = length_dict[domain_slot] * [1] + \
            (mentioned_slot_pool_size - length_dict[domain_slot]) * [0]
    return updated_mentioned_slot_dict, updated_mentioned_mask_dict, updated_str_mentioned_slot_dict


def predict_label_reconstruct(utterance, mentioned_slots, predict_hit_type, predict_mentioned_slot, hit_value_predict,
                              domain_slot, slot_index_value_dict):
    if predict_hit_type == 0:  # for
        reconstructed_label = 'none'
    elif predict_hit_type == 1:
        reconstructed_label = 'dontcare'
    elif predict_hit_type == 2:
        slot_value = mentioned_slots[predict_mentioned_slot][4]
        mentioned_value = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(slot_value)).strip()
        reconstructed_label = mentioned_value
    elif predict_hit_type == 3:
        if domain_slot_type_map[domain_slot] == 'classify':
            if hit_value_predict == len(slot_index_value_dict[domain_slot]):
                reconstructed_label = 'none'
            else:
                reconstructed_label = slot_index_value_dict[domain_slot][hit_value_predict]
        else:
            assert domain_slot_type_map[domain_slot] == 'span'
            start_idx, end_idx = hit_value_predict
            if start_idx <= end_idx:
                target_utterance = utterance[start_idx: end_idx + 1]
                target_value = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(target_utterance))
                reconstructed_label = target_value
            else:
                reconstructed_label = 'none'
    else:
        raise ValueError('invalid value')
    return reconstructed_label
