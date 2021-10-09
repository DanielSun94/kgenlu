import numpy as np
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
    for slot_name in batch['gate']:
        label_gate[slot_name] = batch['gate'][slot_name]
        predicted_gate_index[slot_name] = predicted_gate[slot_name].argmax(dim=1)
        label_value[slot_name] = batch['label'][slot_name]

        # prediction reorganize
        if slot_type_dict[slot_name] == 'classify':
            predicted_value[slot_name] = predict_dict[slot_name].argmax(dim=1)
        else:
            start_index_predict = predict_dict[slot_name][:, :, 0].argmax(dim=0).unsqueeze(1)
            end_index_predict = predict_dict[slot_name][:, :, 1].argmax(dim=0).unsqueeze(1)
            predicted_value[slot_name] = torch.cat((start_index_predict, end_index_predict), dim=1)

    # performance eval part
    # slot-specific
    evaluation_dict = {}
    for slot_name in predicted_value:
        name = '2_slot_'+slot_name
        evaluation_dict[name] = {'exist_true': [], 'exist_false': [], 'not_exist_true': [], 'not_exist_false': []}
        for index in range(len(label_value[slot_name])):
            label = label_value[slot_name][index]
            predict = predicted_value[slot_name][index]
            et, ef, net, nef = 0, 0, 0, 0
            if slot_type_dict[slot_name] == 'classify':
                label, predict = int(label), int(predict)
                if label != -1:
                    if label == predict:
                        et = 1
                    else:
                        ef = 1
                else:
                    if label == predict:
                        net = 1
                    else:
                        nef = 1
            else:
                label, predict = [int(label[0]), int(label[1])], [int(predict[0]), int(predict[1])]
                if label[0] != -1:
                    if label[0] == predict[0] and label[1] == predict[1]:
                        et = 1
                    else:
                        ef = 1
                else:
                    if label[0] == predict[0] and label[1] == predict[1]:
                        net = 1
                    else:
                        nef = 1
            evaluation_dict[name]['exist_true'].append(et)
            evaluation_dict[name]['exist_false'].append(ef)
            evaluation_dict[name]['not_exist_true'].append(net)
            evaluation_dict[name]['not_exist_false'].append(nef)
    # task-specific
    evaluation_dict['0_general'] = [1 for _ in range(args['batch_size'])]
    for slot_name in predicted_value:
        name = '1_task_'+slot_name.split('-')[0]
        if not evaluation_dict.__contains__(name):
            evaluation_dict[name] = [1 for _ in range(args['batch_size'])]
        for index in range(len(label_value[slot_name])):
            label = label_value[slot_name][index]
            predict = predicted_value[slot_name][index]
            if label != predict:
                evaluation_dict[name][index] = 0
                evaluation_dict['0_general'][index] = 0
    return evaluation_dict


def comprehensive_evaluation(evaluation_batch_list, slot_info_dict, data_type, epoch):
    result_dict = {}
    slot_type_dict = slot_info_dict['slot_type_dict']
    for name in evaluation_batch_list[0]:
        if name.__contains__('task') or name.__contains__('general'):
            result_dict[name] = []
        else:
            result_dict[name] = {}
            for key in evaluation_batch_list[0][name]:
                result_dict[name][key] = []
    for batch_result in evaluation_batch_list:
        for name in batch_result:
            if name.__contains__('task') or name.__contains__('general'):
                result_list = batch_result[name]
                for item in result_list:
                    result_dict[name].append(item)
            else:
                for key in batch_result[name]:
                    result_list = batch_result[name][key]
                    for item in result_list:
                        result_dict[name][key].append(item)

    # save result
    data_to_write = [['epoch', epoch]]
    for key in args:
        data_to_write.append([key, args[key]])
    data_to_write.append([data_type])
    data_to_write.append(['name', 'type', 'accuracy', 'exist_true', 'exist_false', 'not_exist_true', 'not_exist_false'])

    data_to_write_ = []
    for name in result_dict:
        name_type = slot_type_dict[name.split('_')[2]] if len(name.split('_'))>2 and name.split('_')[2] in slot_type_dict else ''
        exist_true = str(round(np.sum(result_dict[name]['exist_true'] if isinstance(result_dict[name], dict) else [0]) / len(result_dict['0_general'])*100, 2))+"%"
        exist_false = str(round(np.sum(result_dict[name]['exist_false'] if isinstance(result_dict[name], dict) else [0]) / len(result_dict['0_general'])*100, 2))+"%"
        not_exist_true = str(round(np.sum(result_dict[name]['not_exist_true'] if isinstance(result_dict[name], dict) else [0]) / len(result_dict['0_general'])*100, 2))+"%"
        not_exist_false = str(round(np.sum(result_dict[name]['not_exist_false'] if isinstance(result_dict[name], dict) else [0]) / len(result_dict['0_general'])*100, 2))+"%"
        if result_dict[name].__contains__('exist_true'):
            accuracy = str(round((np.sum(result_dict[name]['exist_true']) + np.sum(result_dict[name]['not_exist_true']))
                                 / len(result_dict[name]['exist_true'])*100, 2))+"%"
        else:
            accuracy = str(round(np.sum(result_dict[name])/len(result_dict[name])))+"%"
        data_to_write_.append([name, name_type, accuracy, exist_true, exist_false, not_exist_true, not_exist_false])

    data_to_write_ = sorted(data_to_write_, key=lambda x: x[0])
    for line in data_to_write_:
        data_to_write.append(line)
    print('epoch: {}'.format(epoch))
    print(['name', 'type', 'accuracy', 'exist_true', 'exist_false', 'not_exist_true', 'not_exist_false'])
    for line in data_to_write_:
        print(line)
    with open(args['evaluation_save_path'], 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(data_to_write)
    return result_dict
