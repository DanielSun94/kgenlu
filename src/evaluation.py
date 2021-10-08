import torch
import csv
from multiwoz_config import args


def evaluation_batch(predicted_gate, predict_dict, batch, slot_info_dict):
    # reorganize part
    label_gate = {}
    predicted_gate_index = {}
    label_value = {}
    predicted_value = {}
    slot_type_dict = slot_info_dict['slot_type_dict']
    for slot_name in batch['gate'][0]:
        label_gate[slot_name] = [item[slot_name] for item in batch['gate']]
        predicted_gate_index[slot_name] = predicted_gate[slot_name].argmax(dim=1)
        label_value[slot_name] = []
        # label reorganize
        for item in batch['label'][slot_name]:
            if slot_type_dict[slot_name] == 'classify':
                if item[0] == -1:
                    label_value[slot_name].append(-1)
                else:
                    label_value[slot_name].append(item[0])
            else:
                if item[0] == -1:
                    label_value[slot_name].append([-1, -1])
                else:
                    label_value[slot_name].append((item[0], item[1]))

        # prediction reorganize
        if slot_type_dict[slot_name] == 'classify':
            predicted_value[slot_name] = predict_dict[slot_name].argmax(dim=1)
        else:
            start_index_predict = predict_dict[slot_name][:, :, 0].argmax(dim=0).unsqueeze(1)
            end_index_predict = predict_dict[slot_name][:, :, 1].argmax(dim=0).unsqueeze(1)
            predicted_value[slot_name] = torch.cat((start_index_predict, end_index_predict), dim=1)

    # performance eval part
    evaluation_dict = {}
    for slot_name in predicted_value:
        evaluation_dict[slot_name] = {'exist': 0, 'match': 0}
        for index in range(len(label_value[slot_name])):
            label = label_value[slot_name][index]
            predict = predicted_value[slot_name][index]
            if slot_type_dict[slot_name] == 'classify':
                assert isinstance(label, int)
                if label != -1:
                    evaluation_dict[slot_name]['exist'] += 1
                if label == predict:
                    evaluation_dict[slot_name]['match'] += 1
            else:
                if label[0] != -1:
                    evaluation_dict[slot_name]['exist'] += 1
                if label[0] == predict[0] and label[1] == predict[1]:
                    evaluation_dict[slot_name]['match'] += 1
    return evaluation_dict


def comprehensive_evaluation(evaluation_batch_list, slot_info_dict, data_type):
    result_dict = {}
    for slot_name in evaluation_batch_list[0]:
        result_dict[slot_name] = {'exist': 0, 'match': 0}
    for batch_result in evaluation_batch_list:
        for slot_name in batch_result:
            result_dict[slot_name]['exist'] += batch_result[slot_name]['exist']
            result_dict[slot_name]['match'] += batch_result[slot_name]['match']

    # save result
    data_to_write = [[data_type], ['slot_name', 'slot_type', 'case number', 'match number', 'accuracy']]
    for slot_name in result_dict:
        case_num = result_dict[slot_name]['exist']
        match_num = result_dict[slot_name]['match']
        if case_num == 0:
            accuracy = "NA"
        else:
            accuracy = str(round(match_num/case_num*100, 2))+"%"
        data_to_write.append([slot_name, slot_info_dict[slot_name], case_num, match_num, accuracy])
    with open(args['evaluation_save_path'], 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(data_to_write)
    return result_dict
