import torch
from torch import BoolTensor, LongTensor
import os
from history_read_data import prepare_data, domain_slot_list, domain_slot_type_map, SampleDataset
from hisory_model import HistorySelectionModel
from history_config import args, DEVICE, medium_result_template, evaluation_folder, ckpt_template, logger
import pickle
import torch.multiprocessing as mp
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from history_evaluation import reconstruct_batch_predict_label_train, batch_eval, comprehensive_eval,\
    evaluation_test_eval
from transformers import get_linear_schedule_with_warmup, AdamW


PROCESS_GLOBAL_NAME = args['process_name']
use_multi_gpu = args['multi_gpu']
overwrite = args['overwrite_cache']
start_epoch = args['start_epoch']
predefined_ckpt_path = args['load_ckpt_path']
mode = args['mode']
train_epoch = args['epoch']
warmup_proportion = args['warmup_proportion']
lr = args['learning_rate']
adam_epsilon = args['adam_epsilon']
max_grad_norm = args['max_grad_norm']
mentioned_slot_pool_size = args['mentioned_slot_pool_size']
pretrained_model = args['pretrained_model']
gate_weight = float(args['gate_weight'])
span_weight = float(args['span_weight'])
classify_weight = float(args['classify_weight'])
mentioned_weight = float(args['mentioned_weight'])
weight_decay = args['weight_decay']


def train(model, name, train_loader, dev_loader, test_loader, classify_slot_index_value_dict,
          classify_slot_value_index_dict, local_rank=None):
    max_step = len(train_loader) * train_epoch
    num_warmup_steps = int(len(train_loader) * train_epoch * warmup_proportion)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=max_step)
    global_step = 0
    current_process_ckpt_path = None
    for epoch in range(train_epoch):
        logger.info("Epoch :{}".format(epoch))
        if use_multi_gpu:
            train_loader.sampler.set_epoch(epoch)

        epoch_result = []
        # Run the train function
        if mode == 'train':
            model.train()
            full_loss = 0
            for train_batch in tqdm(train_loader):
                global_step += 1
                if global_step < start_epoch * len(train_loader):
                    scheduler.step()
                    continue

                if not use_multi_gpu:
                    train_batch = data_device_alignment(train_batch, DEVICE)
                else:
                    train_batch = data_device_alignment(train_batch, local_rank)
                predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict = model(train_batch)
                loss, train_batch_predict_label_dict = train_compute_loss_and_batch_eval(
                    predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict, train_batch,
                    classify_slot_index_value_dict, local_rank)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                full_loss += loss.detach().item()
                epoch_result.append(batch_eval(train_batch_predict_label_dict, train_batch))
                # for possible CUDA out of memory
                del loss, predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict, train_batch
            logger.info('average loss of epoch: {}: {}'.format(epoch, full_loss / len(train_loader)))

            if use_multi_gpu:
                file_path = medium_result_template.format('train', PROCESS_GLOBAL_NAME, epoch, local_rank)
                pickle.dump(epoch_result, open(file_path, 'wb'))
                torch.distributed.barrier()
                if local_rank == 0:
                    result_list = load_result_multi_gpu('train', epoch)
                    result_print(comprehensive_eval(result_list, 'train', PROCESS_GLOBAL_NAME, epoch))
                torch.distributed.barrier()
            else:
                result_print(comprehensive_eval(epoch_result, 'train', PROCESS_GLOBAL_NAME, epoch))

            # save model
            if (use_multi_gpu and local_rank == 0) or not use_multi_gpu:
                current_process_ckpt_path = ckpt_template.format(PROCESS_GLOBAL_NAME, epoch)
                save_model(use_multi_gpu, model, current_process_ckpt_path, local_rank)
            if use_multi_gpu:
                torch.distributed.barrier()

        # validation and test，此处因为原始数据需要顺序输入，写多卡会非常麻烦，因此只使用单卡
        if (use_multi_gpu and local_rank == 0) or not use_multi_gpu:
            if use_multi_gpu:
                assert local_rank is not None
                rank = local_rank
                target_device = local_rank
            else:
                rank = None
                target_device = DEVICE

            if mode != 'train':
                assert current_process_ckpt_path is None and predefined_ckpt_path is not None
                eval_model = HistorySelectionModel(name, pretrained_model, classify_slot_value_index_dict,
                                                   local_rank=rank)
                eval_model = eval_model.cuda(target_device)
                eval_model.get_common_token_embedding(classify_slot_value_index_dict)
                load_model(multi_gpu=use_multi_gpu, model=eval_model, ckpt_path=predefined_ckpt_path,
                           local_rank=rank)
            else:
                assert current_process_ckpt_path is not None
                eval_model = HistorySelectionModel(name, pretrained_model, classify_slot_value_index_dict,
                                                   local_rank=rank)
                eval_model = eval_model.cuda(target_device)
                eval_model.get_common_token_embedding(classify_slot_value_index_dict)
                load_model(multi_gpu=use_multi_gpu, model=eval_model, ckpt_path=current_process_ckpt_path,
                           local_rank=rank)
            logger.info('start evaluation in dev dataset, epoch: {}'.format(epoch))
            model_eval(eval_model, dev_loader, 'dev', epoch, classify_slot_index_value_dict, target_device)
            logger.info('start evaluation in test dataset, epoch: {}'.format(epoch))
            model_eval(eval_model, test_loader, 'test', epoch, classify_slot_index_value_dict, target_device)

        if use_multi_gpu:
            logger.info('epoch finished, process: {}, before barrier'.format(local_rank))
            torch.distributed.barrier()
            logger.info('epoch finished, process: {}, after barrier'.format(local_rank))


def save_model(multi_gpu, model, ckpt_path, local_rank=None):
    if multi_gpu:
        if local_rank == 0:
            torch.save(model.state_dict(), ckpt_path)
        torch.distributed.barrier()
    else:
        torch.save(model.state_dict(), ckpt_path)
    logger.info('save model success')


def load_model(multi_gpu, model, ckpt_path, local_rank=None):
    if multi_gpu:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        state_dict = torch.load(ckpt_path, map_location=map_location)
        new_state_dict = OrderedDict()
        for key in state_dict:
            if 'module.' in key:
                new_state_dict[key.replace('module.', '')] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        model.load_state_dict(new_state_dict)
    else:
        assert local_rank is None
        state_dict = torch.load(ckpt_path, map_location=torch.device(DEVICE))
        new_state_dict = OrderedDict()
        for key in state_dict:
            if 'module.' in key:
                new_state_dict[key.replace('module.', '')] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        model.load_state_dict(new_state_dict)
    logger.info('load model success')


def data_device_alignment(batch, target_device):
    batch = list(batch)
    # 0 sample id, 1 active domain, 2 active slot, 3 context, 4 context mask, 5 true label, 6 hit type,
    # 7 mentioned_id, 8 hit value, 9 mentioned slot, 10 mentioned slot mask
    batch[1] = batch[1].to(target_device)
    batch[2] = batch[2].to(target_device)
    batch[3] = batch[3].to(target_device)
    batch[4] = batch[4].to(target_device)
    for key in batch[5]:
        batch[6][key] = batch[6][key].to(target_device)
        batch[7][key] = batch[7][key].to(target_device)
        batch[10][key] = batch[10][key].to(target_device)
    return batch


def result_print(comprehensive_result):
    for line in comprehensive_result:
        logger.info(line)


def train_compute_loss_and_batch_eval(predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict, train_batch,
                                      classify_slot_index_value_dict, local_rank=None):
    # 0 sample id, 1 active domain, 2 active slot, 3 context, 4 context mask, 5 true label, 6 hit type,
    # 7 mentioned_id, 8 hit value, 9 mentioned slot, 10 mentioned slot mask
    batch_predict_label_dict = {}
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1).to(DEVICE)
    if local_rank is not None:
        cross_entropy = cross_entropy.cuda(local_rank)
    gate_loss, classify_loss, mentioned_loss, span_loss = 0, 0, 0, 0
    for domain_slot in domain_slot_list:
        if not use_multi_gpu:
            predict_hit_type_one_slot = predict_gate_dict[domain_slot].to(DEVICE)
            predict_value_one_slot = predict_value_dict[domain_slot].to(DEVICE)
            predict_mentioned_slot = predict_mentioned_slot_dict[domain_slot].to(DEVICE)
            label_hit_type_one_slot = train_batch[6][domain_slot].to(DEVICE)
            label_value_one_slot = train_batch[8][domain_slot].to(DEVICE)
            label_mentioned_slot_id = train_batch[7][domain_slot].to(DEVICE)
        else:
            predict_hit_type_one_slot = predict_gate_dict[domain_slot].to(local_rank)
            predict_value_one_slot = predict_value_dict[domain_slot].to(local_rank)
            predict_mentioned_slot = predict_mentioned_slot_dict[domain_slot].to(local_rank)
            label_hit_type_one_slot = train_batch[6][domain_slot].to(local_rank)
            label_value_one_slot = train_batch[8][domain_slot].to(local_rank)
            label_mentioned_slot_id = train_batch[7][domain_slot].to(local_rank)

        batch_predict_label_dict[domain_slot] = reconstruct_batch_predict_label_train(
            domain_slot, predict_hit_type_one_slot, predict_value_one_slot,
            predict_mentioned_slot, train_batch, classify_slot_index_value_dict)

        gate_loss += cross_entropy(predict_hit_type_one_slot, label_hit_type_one_slot)
        mentioned_loss += cross_entropy(predict_mentioned_slot, label_mentioned_slot_id)
        if domain_slot_type_map[domain_slot] == 'span':
            pred_start, pred_end = predict_value_one_slot[:, :, 0], predict_value_one_slot[:, :, 1]
            label_start, label_end = label_value_one_slot[:, 0], label_value_one_slot[:, 1]
            span_loss += (cross_entropy(pred_start, label_start) + cross_entropy(pred_end, label_end)) / 2
        else:
            assert domain_slot_type_map[domain_slot] == 'classify'
            classify_loss += cross_entropy(predict_value_one_slot, label_value_one_slot)
    loss = gate_weight*gate_loss + classify_weight*classify_loss + mentioned_weight*mentioned_loss + \
        span_weight*span_loss
    return loss, batch_predict_label_dict


def model_eval(model, data_loader, data_type, epoch, classify_slot_index_value_dict, local_rank=None):
    # eval的特点在于data_loader顺序采样
    model.eval()
    result_list = []
    last_mentioned_slot_dict, last_mentioned_mask_dict, last_str_mentioned_slot_dict, last_sample_id = {}, {}, {}, ''
    with torch.no_grad():
        if (use_multi_gpu and local_rank == 0) or (not use_multi_gpu):
            for batch in tqdm(data_loader):
                if not use_multi_gpu:
                    batch = data_device_alignment(batch, DEVICE)
                else:
                    batch = data_device_alignment(batch, local_rank)

                # 当id 更新时，需要提前重置last sample id
                current_sample_id = batch[0][0].lower().split('.json')[0].strip()
                if current_sample_id != last_sample_id:
                    last_sample_id = current_sample_id
                    last_mentioned_slot_dict, last_mentioned_mask_dict, last_str_mentioned_slot_dict = {}, {}, {}
                    for domain_slot in domain_slot_list:
                        last_mentioned_slot_dict[domain_slot] = [[[1], [1], [1], [1], [1]] for _ in
                                                                 range(mentioned_slot_pool_size)]
                        last_mentioned_mask_dict[domain_slot] = [True] + (mentioned_slot_pool_size - 1) * [False]
                        last_str_mentioned_slot_dict[domain_slot] = \
                            [['<pad>', '<pad>', '<pad>', '<pad>', '<pad>'] for _ in range(mentioned_slot_pool_size)]

                # 因为模型的设置原因，测试阶段batch长度必须为1，不能像Trippy一样做batch测试
                # 在测试时，需要将batch中的mentioned slots真值替换为为模型预测值，以确保测试公平性，也即是替换batch中的9,10,11项
                assert len(batch[0]) == 1
                for domain_slot in domain_slot_list:
                    if not use_multi_gpu:
                        batch[10][domain_slot] = BoolTensor([last_mentioned_mask_dict[domain_slot]]).to(DEVICE)
                    else:
                        batch[10][domain_slot] = BoolTensor([last_mentioned_mask_dict[domain_slot]]).to(local_rank)
                    batch[9][domain_slot] = [last_mentioned_slot_dict[domain_slot]]
                    batch[11][domain_slot] = [last_str_mentioned_slot_dict[domain_slot]]

                predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict = model(batch)
                batch_predict_label_dict, last_sample_id, last_mentioned_slot_dict, last_mentioned_mask_dict, \
                    last_str_mentioned_slot_dict = evaluation_test_eval(
                        predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict, batch,
                        classify_slot_index_value_dict, last_mentioned_slot_dict, last_sample_id,
                        last_mentioned_mask_dict, last_str_mentioned_slot_dict)
                result_list.append(batch_eval(batch_predict_label_dict, batch))
            result_print(comprehensive_eval(result_list, data_type, PROCESS_GLOBAL_NAME, epoch))
    logger.info('model eval, data: {}, epoch: {} finished'.format(data_type, epoch))


def load_result_multi_gpu(data_type, epoch):
    file_list, target_file_list = os.listdir(evaluation_folder), []
    key_name = (data_type + '_' + PROCESS_GLOBAL_NAME + '_' + str(epoch)).strip()
    for file_name in file_list:
        if key_name in file_name:
            target_file_list.append(file_name)
    assert len(target_file_list) == torch.cuda.device_count()
    result_list = []
    for file_name in target_file_list:
        batch_result = pickle.load(open(os.path.join(evaluation_folder, file_name), 'rb'))
        for sample_result in batch_result:
            result_list.append(sample_result)
    return result_list


def single_gpu_main(pass_info):
    classify_slot_value_index_dict, classify_slot_index_value_dict, train_loader, dev_loader, test_loader = \
        prepare_data(overwrite=overwrite)
    pretrained_model, name = pass_info
    model = HistorySelectionModel(name, pretrained_model, classify_slot_value_index_dict)
    model = model.cuda(DEVICE)
    # 注意，这一步必须在模型置cuda后手工立即进行
    model.get_common_token_embedding(classify_slot_value_index_dict)
    if os.path.exists(predefined_ckpt_path):
        load_model(use_multi_gpu, model, predefined_ckpt_path)
    train(model, name, train_loader, dev_loader, test_loader, classify_slot_index_value_dict,
          classify_slot_value_index_dict)


def multi_gpu_main(local_rank, _, pass_info):
    pretrained_model, name = pass_info
    num_gpu = torch.cuda.device_count()
    logger.info('GPU count: {}'.format(num_gpu))
    torch.distributed.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:23456', world_size=num_gpu,
                                         rank=local_rank)
    classify_slot_value_index_dict, classify_slot_index_value_dict, train_loader, dev_loader, test_loader = \
        prepare_data(overwrite=overwrite)
    logger.info('world size: {}'.format(torch.distributed.get_world_size()))
    local_rank = torch.distributed.get_rank()
    logger.info('local rank: {}'.format(local_rank))
    # DEVICE = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    model = HistorySelectionModel(name, pretrained_model, classify_slot_value_index_dict, local_rank)
    model = model.cuda(local_rank)  # 将模型拷贝到每个gpu上
    # 注意，这一步必须在模型置cuda后手工立即进行
    model.get_common_token_embedding(classify_slot_value_index_dict)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)

    if os.path.exists(predefined_ckpt_path):
        load_model(use_multi_gpu, model, predefined_ckpt_path, local_rank)

    train(model, name, train_loader, dev_loader, test_loader, classify_slot_index_value_dict,
          classify_slot_value_index_dict, local_rank)


def main():
    pass_info = pretrained_model, PROCESS_GLOBAL_NAME
    logger.info('start training')
    if use_multi_gpu:
        num_gpu = torch.cuda.device_count()
        mp.spawn(multi_gpu_main, nprocs=num_gpu, args=(num_gpu, pass_info))
    else:
        single_gpu_main(pass_info)


if __name__ == '__main__':
    args_list = []
    for item in args:
        args_list.append([item, args[item]])
    args_list = sorted(args_list, key=lambda x: x[0])
    for item in args_list:
        logger.info('{} value: {}'.format(item[0], item[1]))
    main()
