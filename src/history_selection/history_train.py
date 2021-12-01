import logging
import torch
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
    evaluation_test_batch_eval
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup, AdamW


PROCESS_GLOBAL_NAME = args['process_name']
use_multi_gpu = args['multi_gpu']
overwrite = args['overwrite_cache']
start_epoch = args['start_epoch']
load_ckpt_path = args['load_ckpt_path']
mode = args['mode']
train_epoch = args['epoch']
warmup_proportion = args['warmup_proportion']
lr = args['learning_rate']
adam_epsilon = args['adam_epsilon']
max_grad_norm = args['max_grad_norm']
mentioned_slot_pool_size = args['mentioned_slot_pool_size']


def train(model, name, train_loader, dev_loader, test_loader, classify_slot_index_value_dict,
          classify_slot_value_index_dict, local_rank=None):
    max_step = len(train_loader) * train_epoch
    num_warmup_steps = int(len(train_loader) * train_epoch * warmup_proportion)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=max_step)
    global_step = 0
    ckpt_path = None
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
                    train_batch = data_device_alignment(train_batch)
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

            #  save model
            ckpt_path = ckpt_template.format(PROCESS_GLOBAL_NAME, epoch)
            save_model(use_multi_gpu, model, ckpt_path, local_rank)

        # validation and test，此处因为原始数据判定的顺序问题，不可以使用distributed model，因此要重新载入
        if (use_multi_gpu and local_rank == 0) or not use_multi_gpu:
            if mode != 'train':
                assert ckpt_path is None and load_ckpt_path is not None
                eval_model = HistorySelectionModel(name, args['pretrained_model'], classify_slot_value_index_dict)
                eval_model = eval_model.cuda(DEVICE)
                eval_model.get_common_token_embedding()
                load_model(multi_gpu=False, model=eval_model, ckpt_path=load_ckpt_path)
            else:
                assert ckpt_path is not None
                eval_model = HistorySelectionModel(name, args['pretrained_model'], classify_slot_value_index_dict)
                eval_model = eval_model.cuda(DEVICE)
                eval_model.get_common_token_embedding()
                load_model(multi_gpu=False, model=eval_model, ckpt_path=ckpt_path)
            logger.info('start evaluation in dev dataset, epoch: {}'.format(epoch))
            model_eval(eval_model, dev_loader, 'dev', epoch, classify_slot_index_value_dict, local_rank)
            logger.info('start evaluation in test dataset, epoch: {}'.format(epoch))
            model_eval(eval_model, test_loader, 'test', epoch, classify_slot_index_value_dict, local_rank)
        if use_multi_gpu:
            torch.distributed.barrier()


def save_model(multi_gpu, model, ckpt_path, local_rank=None):
    if multi_gpu:
        if local_rank == 0:
            torch.save(model.state_dict(), ckpt_path)
        dist.barrier()
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
                new_state_dict[key] = state_dict[key]
            else:
                new_state_dict['module.'+key] = state_dict[key]
        model.load_state_dict(state_dict)
    else:
        state_dict = torch.load(ckpt_path, map_location=torch.device(DEVICE))
        new_state_dict = OrderedDict()
        for key in state_dict:
            if 'module.' in key:
                new_state_dict[key.replace('module.', '')] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        model.load_state_dict(new_state_dict)
    logger.info('load model success')


def data_device_alignment(batch):
    batch = list(batch)
    # 0 sample id, 1 active domain, 2 active slot, 3 context, 4 context mask, 5 true label, 6 hit type,
    # 7 mentioned_id, 8 hit value, 9 mentioned slot, 10 mentioned slot mask
    batch[1] = batch[1].to(DEVICE)
    batch[2] = batch[2].to(DEVICE)
    batch[3] = batch[3].to(DEVICE)
    batch[4] = batch[4].to(DEVICE)
    for key in batch[5]:
        batch[6][key] = batch[6][key].to(DEVICE)
        batch[7][key] = batch[7][key].to(DEVICE)
        batch[10][key] = batch[10][key].to(DEVICE)
    return batch


def result_print(comprehensive_result):
    for line in comprehensive_result:
        logger.info(line)


def train_compute_loss_and_batch_eval(predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict, train_batch,
                                      classify_slot_index_value_dict, local_rank=None):
    # 0 sample id, 1 active domain, 2 active slot, 3 context, 4 context mask, 5 true label, 6 hit type,
    # 7 mentioned_id, 8 hit value, 9 mentioned slot, 10 mentioned slot mask
    gate_weight = float(args['gate_weight'])
    span_weight = float(args['span_weight'])
    classify_weight = float(args['classify_weight'])
    mentioned_weight = float(args['mentioned_weight'])
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
    last_mentioned_slot_dict, last_mentioned_mask_dict, last_sample_id = {}, {}, ''
    for domain_slot in domain_slot_list:
        last_mentioned_slot_dict[domain_slot] = [[[3], [3], [3], [3], [3]] for _ in range(mentioned_slot_pool_size)]
        last_mentioned_mask_dict[domain_slot] = [1] + (mentioned_slot_pool_size - 1) * [0]
    with torch.no_grad():
        if (use_multi_gpu and local_rank == 0) or (not use_multi_gpu):
            for batch in tqdm(data_loader):
                if not use_multi_gpu:
                    batch = data_device_alignment(batch)

                predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict = model(batch)
                batch_predict_label_dict, last_sample_id, last_mentioned_slot_dict, last_mentioned_mask_dict = \
                    evaluation_test_batch_eval(predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict,
                                               batch, classify_slot_index_value_dict, last_mentioned_slot_dict,
                                               last_sample_id, last_mentioned_mask_dict)
                result_list.append(batch_eval(batch_predict_label_dict, batch))
            result_print(comprehensive_eval(result_list, data_type, PROCESS_GLOBAL_NAME, epoch))
    if use_multi_gpu:
        torch.distributed.barrier()
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
    model.get_common_token_embedding()
    if os.path.exists(load_ckpt_path):
        load_model(use_multi_gpu, model, load_ckpt_path)
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
    model.get_common_token_embedding()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)

    if os.path.exists(load_ckpt_path):
        load_model(use_multi_gpu, model, load_ckpt_path, local_rank)

    train(model, name, train_loader, dev_loader, test_loader, classify_slot_index_value_dict,
          classify_slot_value_index_dict, local_rank)


def main():
    pass_info = args['pretrained_model'], args['process_name']
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
