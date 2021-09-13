"""
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
"""
import os
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import nn
import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
import json
from multiwoz_mask import masked_cross_entropy_for_value
from multiwoz_config import USE_CUDA, args, PAD_token
from multiwoz_preprocess import prepare_data_seq, multiwoz_resource_folder, Lang
from model_eval import evaluate


class TRADE(nn.Module):
    def __init__(self, hidden_size, lang, path, lr, dropout, slots, gating_dict):
        super(TRADE, self).__init__()
        self.name = "TRADE"
        self.hidden_size = hidden_size
        self.lang = lang[0]
        self.mem_lang = lang[1]
        self.lr = lr
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.loss_grad = None
        self.loss_ptr_to_bp = None
        self.loss = None
        self.print_every = None
        self.loss_ptr = None
        self.loss_gate = None
        self.loss_class = None
        self.copy_list = None

        self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout,
                                 self.slots, self.nb_gate)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path) + '/enc.th')
                trained_decoder = torch.load(str(path) + '/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                trained_decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)

            self.encoder.load_state_dict(trained_encoder.state_dict())
            self.decoder.load_state_dict(trained_decoder.state_dict())

        # Initialize optimizers and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        # print_loss_class = self.loss_class / self.print_every
        # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg, print_loss_ptr, print_loss_gate)

    def save_model(self, dec_type):
        directory = os.path.join(multiwoz_resource_folder, 'save/TRADE-' + args["add_name"] + '/' + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch_size']) + 'DR' + str(self.dropout) + str(dec_type))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data_, _, slot_temp, reset=0):
        if reset:
            self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]
        all_point_outputs, gates, words_point_out, words_class_out = self.encode_and_decode(data_, use_teacher_forcing,
                                                                                            slot_temp)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data_["generate_y"].contiguous(),  # [:,:len(self.point_slots)].contiguous(),
            data_["y_lengths"])  # [:,:len(self.point_slots)])
        loss_gate = self.cross_entropy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
                                       data_["gating_label"].contiguous().view(-1))

        if args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr

        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

    def optimize(self, clip):
        self.loss_grad.backward()
        # clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def optimize_GEM(self, clip):
        # clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def encode_and_decode(self, data_, use_teacher_forcing, slot_temp):
        # Build unknown mask for memory to encourage generalization
        if args['unk_mask'] and self.decoder.training:
            story_size = data_['context'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)
            if USE_CUDA:
                rand_mask = rand_mask.cuda()
            story = data_['context'] * rand_mask.long()
        else:
            story = data_['context']

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data_['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data_['context_len'])
        self.copy_list = data_['context_plain']
        max_res_len = data_['generate_y'].size(2) if self.encoder.training else 10
        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = \
            self.decoder.forward(batch_size, encoded_hidden, encoded_outputs, data_['context_len'], story, max_res_len,
                                 data_['generate_y'], use_teacher_forcing, slot_temp)
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def evaluate(self, dev, slot_temp):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            _, gates, words, class_words = self.encode_and_decode(data_dev, False, slot_temp)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {
                    "turn_belief": data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                if args["use_gate"]:
                    for si, sg in enumerate(gate):
                        if sg == self.gating_dict["none"]:
                            continue
                        elif sg == self.gating_dict["ptr"]:
                            pred = np.transpose(words[si])[bi]
                            st = []
                            for e in pred:
                                if e == 'EOS':
                                    break
                                else:
                                    st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + inverse_unpoint_slot[sg.item()])
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS':
                                break
                            else:
                                st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr
                if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                    print("True", set(data_dev["turn_belief"][bi]))
                    print("Pred", set(predict_belief_bsz_ptr), "\n")

        self.encoder.train(True)
        self.decoder.train(True)
        return all_prediction


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        # encoder开始处载入embedding
        if args["load_embedding"]:
            with open(os.path.join(multiwoz_resource_folder, 'emb{}.json'.format(vocab_size))) as f:
                e = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(e))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    def get_state(self, bsz):
        """
        hidden state (zero) initialization
        Get cell states and hidden states.
        """
        if USE_CUDA:
            return Variable(torch.zeros(2, bsz, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, _=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)

        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches,
                use_teacher_forcing, slot_temp):
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate)
        if USE_CUDA:
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()

        # Get the slot embedding
        # 给30个slot pair每个一个embedding
        # slot是一个domain-slot结构，因此split之后一定有且只有两个element
        # 其实这段代码可以优化（slot temp list在一个任务里是不变的，没必要每次调用都算一遍），凡是先放着不管吧
        slot_emb_dict = {}
        slot_emb_arr = None
        for i, slot in enumerate(slot_temp):
            # Domain embedding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                if USE_CUDA:
                    domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            else:
                raise ValueError('Slot embedding Value Error')
            # Slot embedding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                if USE_CUDA:
                    slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)
            else:
                raise ValueError('Slot embedding Value Error')

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        if args["parallel_decode"]:
            # Compute pointer-generator output, putting all (domain, slot) in one batch
            decoder_input = self.dropout_layer(slot_emb_arr).view(-1, self.hidden_size)  # (batch*|slot|) * emb
            hidden = encoded_hidden.repeat(1, len(slot_temp), 1)  # 1 * (batch*|slot|) * emb
            words_point_out = [[] for _ in range(len(slot_temp))]
            # words_class_out = []

            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
                enc_len = encoded_lens * len(slot_temp)
                context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

                if wi == 0:
                    all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())

                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_ptr = torch.zeros(p_vocab.size())
                if USE_CUDA:
                    p_context_ptr = p_context_ptr.cuda()

                p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words = [self.lang.index2word[w_idx.item()] for w_idx in pred_word]

                for si in range(len(slot_temp)):
                    words_point_out[si].append(words[si * batch_size:(si + 1) * batch_size])

                all_point_outputs[:, :, wi, :] = torch.reshape(final_p_vocab,
                                                               (len(slot_temp), batch_size, self.vocab_size))

                if use_teacher_forcing:
                    decoder_input = self.embedding(torch.flatten(target_batches[:, :, wi].transpose(1, 0)))
                else:
                    decoder_input = self.embedding(pred_word)

                if USE_CUDA:
                    decoder_input = decoder_input.cuda()
        else:
            # Compute pointer-generator output, decoding each (domain, slot) one-by-one
            words_point_out = []
            counter = 0
            for slot in slot_temp:
                hidden = encoded_hidden
                words = []
                slot_emb = slot_emb_dict[slot]
                decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)
                for wi in range(max_res_len):
                    dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                    context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                    if wi == 0:
                        # gate output 只在一开始算一次。并且和序列输出独立
                        all_gate_outputs[counter] = self.W_gate(context_vec)
                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                    p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                    p_context_ptr = torch.zeros(p_vocab.size())
                    if USE_CUDA:
                        p_context_ptr = p_context_ptr.cuda()
                    p_context_ptr.scatter_add_(1, story, prob)
                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                        vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                    pred_word = torch.argmax(final_p_vocab, dim=1)
                    words.append([self.lang.index2word[w_idx.item()] for w_idx in pred_word])
                    all_point_outputs[counter, :, wi, :] = final_p_vocab
                    if use_teacher_forcing:
                        decoder_input = self.embedding(target_batches[:, counter, wi])  # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_word)
                    if USE_CUDA:
                        decoder_input = decoder_input.cuda()
                counter += 1
                words_point_out.append(words)

        return all_point_outputs, all_gate_outputs, words_point_out, []

    @staticmethod
    def attend(seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    @staticmethod
    def attend_vocab(seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


def main():
    early_stop = args['earlyStop']

    # Configure models and load data
    avg_best, cnt, acc = 0.0, 0, 0.0
    train, dev, test, test_special, lang, slots_list, gating_dict, max_word = \
        prepare_data_seq(True, batch_size=int(args['batch_size']))
    assert isinstance(lang[0], Lang)
    assert isinstance(lang[1], Lang)
    model = TRADE(hidden_size=int(args['hidden']), lang=lang, path=args['path'], lr=float(args['learn']),
                  dropout=float(args['drop']), slots=slots_list, gating_dict=gating_dict)

    # print("[Info] Slots include ", SLOTS_LIST)
    # print("[Info] Unpointable Slots include ", gating_dict)

    for epoch in range(args['epoch']):
        print("Epoch :{}".format(epoch))
        # Run the train function
        p_bar = tqdm(enumerate(train), total=len(train))
        for i, data in p_bar:
            model.train_batch(data, int(args['clip']), slots_list[1], reset=(i == 0))
            model.optimize(args['clip'])
            p_bar.set_description(model.print_loss())
            # print(data)
            # exit(1)

        if (epoch + 1) % int(args['evalp']) == 0:
            all_prediction = model.evaluate(dev, slots_list[2])
            joint_acc_score, f1_score = evaluate(all_prediction, model.name, slots_list[2])

            if early_stop == 'F1':
                if f1_score >= avg_best:
                    model.save_model('ENTF1-{:.4f}'.format(f1_score))
                    print("MODEL SAVED")
                return f1_score
            else:
                if joint_acc_score >= avg_best:
                    model.save_model('ACC-{:.4f}'.format(joint_acc_score))
                    print("MODEL SAVED")

            model.scheduler.step(acc)

            if acc >= avg_best:
                avg_best = acc
                cnt = 0
                # best_model = model
            else:
                cnt += 1

            if cnt == args["patience"] or (acc == 1.0 and early_stop is None):
                print("Ran out of patient, early stop...")
                break
    all_prediction = model.evaluate(dev, slots_list[2])
    _, __ = evaluate(all_prediction, model.name, slots_list[2])


if __name__ == '__main__':
    main()
