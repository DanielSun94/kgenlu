import logging
import torch
import os
from kgenlu_read_data import prepare_data, Sample, domain_slot_list, domain_slot_type_map
from kgenlu_model import KGENLU
from kgenlu_config import args, logger, DEVICE, medium_result_template, evaluation_folder, ckpt_template
import pickle
import torch.multiprocessing as mp
from torch import nn, optim
from tqdm import tqdm
from kgenlu_evaluation import reconstruct_batch_predict_label, batch_eval, comprehensive_eval
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup, AdamW

PROCESS_GLOBAL_NAME = 'no4'
use_multi_gpu = args['multi_gpu']
load_cpkt_name = ''  # os.path.join(os.path.abspath('../../resource/model_checkpoint'), 'no1_0.ckpt')
start_epoch = 0
overwrite = False


def train(model, train_loader, dev_loader, test_loader, classify_slot_index_value_dict, local_rank=None):
    max_step = len(train_loader) * args['epoch']
    num_warmup_steps = int(len(train_loader) * args['epoch'] * args['warmup_proportion'])
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=max_step)
    global_step = 0

    for epoch in range(args['epoch']):
        logger.info("Epoch :{}".format(epoch))
        if use_multi_gpu:
            train_loader.sampler.set_epoch(epoch)

        epoch_result = []
        # Run the train function
        model.train()
        full_loss = 0
        for train_batch in tqdm(train_loader):
            global_step += 1
            if global_step < start_epoch * len(train_loader):
                scheduler.step()
                continue

            if not use_multi_gpu:
                train_batch = data_device_alignment(train_batch)
            predict_gate, predict_dict, referred_dict = model(train_batch)
            loss, train_batch_predict_label_dict = \
                compute_loss_and_batch_eval(predict_gate, predict_dict, referred_dict, train_batch,
                                            classify_slot_index_value_dict, local_rank)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            full_loss += loss.detach().item()
            epoch_result.append(batch_eval(train_batch_predict_label_dict, train_batch))
            del loss, predict_gate, predict_dict, referred_dict, train_batch  # for possible CUDA out of memory

        if use_multi_gpu:
            pickle.dump(epoch_result,
                        open(medium_result_template.format('train', PROCESS_GLOBAL_NAME, epoch, local_rank), 'wb'))
            torch.distributed.barrier()
            if local_rank == 0:
                result_list = load_result_multi_gpu('train', epoch)
                result_print(comprehensive_eval(result_list, 'train', PROCESS_GLOBAL_NAME,  epoch))
            torch.distributed.barrier()
        else:
            result_print(comprehensive_eval(epoch_result, 'train', PROCESS_GLOBAL_NAME, epoch))

        #  save model
        ckpt_path = ckpt_template.format(PROCESS_GLOBAL_NAME, epoch)
        save_model(use_multi_gpu, model, ckpt_path, local_rank)

        # validation and test
        logger.info('start evaluation in dev dataset, epoch: {}'.format(epoch))
        model_eval(model, dev_loader, 'dev', epoch, classify_slot_index_value_dict, local_rank)
        logger.info('start evaluation in test dataset, epoch: {}'.format(epoch))
        model_eval(model, test_loader, 'test', epoch, classify_slot_index_value_dict, local_rank)


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
        model.load_state_dict(
            torch.load(ckpt_path, map_location=map_location))
    else:
        model.load_state_dict(
            torch.load(ckpt_path))
    logger.info('load model success')


def data_device_alignment(batch):
    batch = list(batch)
    batch[1] = batch[1].to(DEVICE)
    batch[2] = batch[2].to(DEVICE)
    batch[5] = batch[5].to(DEVICE)
    batch[9] = batch[9].to(DEVICE)
    for key in batch[6]:
        batch[6][key] = batch[6][key].to(DEVICE)
        batch[7][key] = batch[7][key].to(DEVICE)
        batch[8][key] = batch[8][key].to(DEVICE)
    return batch


def result_print(comprehensive_result):
    for line in comprehensive_result:
        print(line)


def compute_loss_and_batch_eval(predict_gate, predict_dict, referred_dict, train_batch, classify_slot_index_value_dict,
                                local_rank=None):
    gate_weight = float(args['gate_weight'])
    span_weight = float(args['span_weight'])
    classify_weight = float(args['classify_weight'])
    referral_weight = float(args['referral_weight'])
    batch_predict_label_dict = {}
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1).to(DEVICE)
    if local_rank is not None:
        cross_entropy = cross_entropy.cuda(local_rank)
    gate_loss, classify_loss, referral_loss, span_loss = 0, 0, 0, 0
    for domain_slot in domain_slot_list:
        if not use_multi_gpu:
            predict_hit_type_one_slot = predict_gate[domain_slot].to(DEVICE)
            predict_value_one_slot = predict_dict[domain_slot].to(DEVICE)
            predict_referral_one_slot = referred_dict[domain_slot].to(DEVICE)
            label_hit_type_one_slot = train_batch[7][domain_slot].to(DEVICE)
            label_value_one_slot = train_batch[8][domain_slot].to(DEVICE)
            label_referral_one_slot = train_batch[6][domain_slot].to(DEVICE)
        else:
            predict_hit_type_one_slot = predict_gate[domain_slot].to(local_rank)
            predict_value_one_slot = predict_dict[domain_slot].to(local_rank)
            predict_referral_one_slot = referred_dict[domain_slot].to(local_rank)
            label_hit_type_one_slot = train_batch[7][domain_slot].to(local_rank)
            label_value_one_slot = train_batch[8][domain_slot].to(local_rank)
            label_referral_one_slot = train_batch[6][domain_slot].to(local_rank)

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
    loss = gate_weight*gate_loss + classify_weight*classify_loss + referral_weight*referral_loss + span_weight*span_loss
    return loss, batch_predict_label_dict


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


def model_eval(model, data_loader, data_type, epoch, classify_slot_index_value_dict, local_rank=None):
    model.eval()
    result_list = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if not use_multi_gpu:
                batch = data_device_alignment(batch)
            predict_gate, predict_dict, referred_dict = model(batch)
            _, dev_batch_predict_label_dict = \
                compute_loss_and_batch_eval(predict_gate, predict_dict, referred_dict, batch,
                                            classify_slot_index_value_dict, local_rank)
            result_list.append(batch_eval(dev_batch_predict_label_dict, batch))
        if use_multi_gpu:
            pickle.dump(result_list, open(medium_result_template.format(
                data_type, PROCESS_GLOBAL_NAME, epoch, local_rank), 'wb'))
            torch.distributed.barrier()
            if local_rank == 0:
                result_list = load_result_multi_gpu(data_type, epoch)
                result_print(comprehensive_eval(result_list, data_type, PROCESS_GLOBAL_NAME, epoch))
        else:
            result_print(comprehensive_eval(result_list, data_type, PROCESS_GLOBAL_NAME, epoch))
    logger.info('model eval, data: {}, epoch: {} finished'.format(data_type, epoch))


def main():
    pretrained_model = args['pretrained_model']
    name = args['name']
    pass_info = name, pretrained_model
    logger.info('start training')
    if use_multi_gpu:
        num_gpu = torch.cuda.device_count()
        mp.spawn(multi_gpu_main, nprocs=num_gpu, args=(num_gpu, pass_info))
    else:
        single_gpu_main(pass_info)


def single_gpu_main(pass_info):
    data, classify_slot_value_index_dict, classify_slot_index_value_dict = prepare_data(overwrite=overwrite)
    train_loader, dev_loader, test_loader = data
    name, pretrained_model = pass_info
    model = KGENLU(name, pretrained_model, classify_slot_value_index_dict)
    model = model.cuda(DEVICE)
    if os.path.exists(load_cpkt_name):
        load_model(use_multi_gpu, model, load_cpkt_name)
    train(model, train_loader, dev_loader, test_loader, classify_slot_index_value_dict)


def multi_gpu_main(local_rank, _, pass_info):
    name, pretrained_model = pass_info

    num_gpu = torch.cuda.device_count()
    logger.info('GPU count: {}'.format(num_gpu))
    torch.distributed.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:23456', world_size=num_gpu,
                                         rank=local_rank)
    data, classify_slot_value_index_dict, classify_slot_index_value_dict = prepare_data(overwrite=overwrite)
    train_loader, dev_loader, test_loader = data
    logger.info('world size: {}'.format(torch.distributed.get_world_size()))
    local_rank = torch.distributed.get_rank()
    logger.info('local rank: {}'.format(local_rank))
    # DEVICE = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    model = KGENLU(name, pretrained_model, classify_slot_value_index_dict)
    model = model.cuda(local_rank)  # 将模型拷贝到每个gpu上

    if os.path.exists(load_cpkt_name):
        load_model(use_multi_gpu, model, load_cpkt_name, local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    train(model, train_loader, dev_loader, test_loader, classify_slot_index_value_dict, local_rank)


if __name__ == '__main__':
    for item in args:
        logging.info('{} value: {}'.format(item, args[item]))
    main()
