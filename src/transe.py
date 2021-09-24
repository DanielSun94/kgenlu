# the source codes of transE are from https://github.com/mklimasz/TransE-PyTorch
from absl import app
from absl import flags
import os
import numpy as np
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard
from collections import Counter
from torch.utils import data
from typing import Dict, Tuple
import torch
from torch import nn
from torch.optim import optimizer
import pickle

FB15K = 'FB15K'
WORDNET = 'WordNet'
dataset = WORDNET  # WordNet FB15K
FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.01, help="Learning rate value.")
flags.DEFINE_integer("seed", default=715, help="Seed value.")
flags.DEFINE_integer("batch_size", default=128, help="Maximum batch size.")
flags.DEFINE_integer("validation_batch_size", default=64, help="Maximum batch size during model validation.")
flags.DEFINE_integer("vector_length", default=100, help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=1.0, help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer("norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epochs", default=4000, help="Number of training epochs.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")
flags.DEFINE_integer("validation_freq", default=10, help="Validate model every X epochs.")
flags.DEFINE_string("checkpoint_path", default="", help="Path to model checkpoint (by default train from scratch).")
flags.DEFINE_string("tensorboard_log_dir", default=os.path.abspath('../resource/runs/'),
                    help="Path for tensorboard log directory.")
flags.DEFINE_string("checkpoint_folder", default=os.path.abspath('../resource/transe_checkpoint/'),
                    help='checkpoint folder')
flags.DEFINE_string("load_checkpoint_file", default=None, help='checkpoint file')
if dataset == FB15K:
    flags.DEFINE_string("dataset_path", default=os.path.abspath('../resource/FB15k/'), help="Path to dataset.")
elif dataset == WORDNET:
    flags.DEFINE_string("dataset_path", default=os.path.abspath('../resource/wordnet/'), help="Path to dataset.")
else:
    raise ValueError('Illegal Dataset')

HITS_AT_1_SCORE = float
HITS_AT_3_SCORE = float
HITS_AT_10_SCORE = float
MRR_SCORE = float
METRICS = Tuple[HITS_AT_1_SCORE, HITS_AT_3_SCORE, HITS_AT_10_SCORE, MRR_SCORE]
Mapping = Dict[str, int]

_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_EPOCH = "epoch"
_STEP = "step"
_BEST_SCORE = "best_score"


class TransE(nn.Module):
    def __init__(self, entity_count, relation_count, device, norm=1, dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim = dim
        self.entities_emb = self._init_entity_emb()
        self.relations_emb = self._init_relation_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def _init_entity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        """Return model losses based on the input.
        :param positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param negative_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        # -1 to avoid nan for OOV vector
        self.entities_emb.weight.data[:-1, :].div_(self.entities_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        assert positive_triplets.size()[1] == 3
        positive_distances = self._distance(positive_triplets)

        assert negative_triplets.size()[1] == 3
        negative_distances = self._distance(negative_triplets)

        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances

    def predict(self, triplets: torch.LongTensor):
        """Calculated dissimilarity score for given triplets.
        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(triplets)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm,
                                                                                                          dim=1)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optim_: optimizer.Optimizer) -> Tuple[int, int, float]:
    """Loads training checkpoint.
    :param checkpoint_path: path to checkpoint
    :param model: model to update state
    :param optim_: optimizer to  update state
    :return tuple of starting epoch id, starting step id, best checkpoint score
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
    optim_.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    start_epoch_id = checkpoint[_EPOCH] + 1
    step = checkpoint[_STEP] + 1
    best_score = checkpoint[_BEST_SCORE]
    return start_epoch_id, step, best_score


def save_checkpoint(model: nn.Module, optim_: optimizer.Optimizer, epoch_id: int, step: int, best_score: float,
                    save_path: str, kg_name: str):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim_.state_dict(),
        _EPOCH: epoch_id,
        _STEP: step,
        _BEST_SCORE: best_score
    }, os.path.join(save_path, kg_name+'_'+str(epoch_id)+'_checkpoint.tar'))


def hit_at_k(predictions: torch.Tensor, ground_truth_idx: torch.Tensor, device: torch.device, k: int = 10) -> int:
    """Calculates number of hits@k.
    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :param device: device on which calculations are taking place
    :param k: number of top K results to be considered as hits
    :return: Hits@K score
    """
    assert predictions.size(0) == ground_truth_idx.size(0)

    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    _, indices = predictions.topk(k=k, largest=False)
    return torch.where(indices == ground_truth_idx, one_tensor, zero_tensor).sum().item()


def cal_mrr(predictions: torch.Tensor, ground_truth_idx: torch.Tensor) -> float:
    """Calculates mean reciprocal rank (MRR) for given predictions and ground truth values.
    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :return: Mean reciprocal rank score
    """
    assert predictions.size(0) == ground_truth_idx.size(0)

    indices = predictions.argsort()
    return (1.0 / (indices == ground_truth_idx).nonzero()[:, 1].float().add(1.0)).sum().item()


def create_mappings(dataset_path: str) -> Tuple[Mapping, Mapping]:
    """Creates separate mappings to indices for entities and relations."""
    # counters to have entities/relations sorted from most frequent
    entity_counter = Counter()
    relation_counter = Counter()
    with open(dataset_path, "r") as f:
        for line in f:
            # -1 to remove newline sign
            head, relation, tail = line[:-1].split("\t")
            entity_counter.update([head, tail])
            relation_counter.update([relation])
    entity2id = {}
    relation2id = {}
    for idx, (mid, _) in enumerate(entity_counter.most_common()):
        entity2id[mid] = idx
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx
    return entity2id, relation2id


class FB15KDataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, data_path: str, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(data_path, "r") as f:
            # data in tuples (head, relation, tail)
            self.data = [line[:-1].split("\t") for line in f]

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)


class WordNetDataset(data.Dataset):
    """Dataset implementation for handling WordNet."""
    def __init__(self, kg_data: Mapping):
        self.data = kg_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)


def test(model: torch.nn.Module, data_generator: torch_data.DataLoader, entities_count: int,
         summary_writer: tensorboard.SummaryWriter, device: torch.device, epoch_id: int, metric_suffix: str,
         ) -> METRICS:
    examples_count = 0.0
    hits_at_1 = 0.0
    hits_at_3 = 0.0
    hits_at_10 = 0.0
    mrr = 0.0

    entity_ids = torch.arange(end=entities_count, device=device).unsqueeze(0)
    for head, relation, tail in data_generator:
        current_batch_size = head.size()[0]

        head, relation, tail = head.to(device), relation.to(device), tail.to(device)
        all_entities = entity_ids.repeat(current_batch_size, 1)
        heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
        tails = tail.reshape(-1, 1).repeat(1, all_entities.size()[1])

        # Check all possible tails
        triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        tails_predictions = model.predict(triplets).reshape(current_batch_size, -1)
        # Check all possible heads
        triplets = torch.stack((all_entities, relations, tails), dim=2).reshape(-1, 3)
        heads_predictions = model.predict(triplets).reshape(current_batch_size, -1)

        # Concat predictions
        predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
        ground_truth_entity_id = torch.cat((tail.reshape(-1, 1), head.reshape(-1, 1)))

        hits_at_1 += hit_at_k(predictions, ground_truth_entity_id, device=device, k=1)
        hits_at_3 += hit_at_k(predictions, ground_truth_entity_id, device=device, k=3)
        hits_at_10 += hit_at_k(predictions, ground_truth_entity_id, device=device, k=10)
        mrr += cal_mrr(predictions, ground_truth_entity_id)

        examples_count += predictions.size()[0]

    hits_at_1_score = hits_at_1 / examples_count * 100
    hits_at_3_score = hits_at_3 / examples_count * 100
    hits_at_10_score = hits_at_10 / examples_count * 100
    mrr_score = mrr / examples_count * 100
    summary_writer.add_scalar('Metrics/Hits_1/' + metric_suffix, hits_at_1_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/Hits_3/' + metric_suffix, hits_at_3_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/Hits_10/' + metric_suffix, hits_at_10_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/MRR/' + metric_suffix, mrr_score, global_step=epoch_id)

    return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score


def main(_):
    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = FLAGS.batch_size
    vector_length = FLAGS.vector_length
    margin = FLAGS.margin
    norm = FLAGS.norm
    learning_rate = FLAGS.lr
    epochs = FLAGS.epochs
    device = torch.device('cuda') if FLAGS.use_gpu else torch.device('cpu')

    path = FLAGS.dataset_path
    train_generator, validation_generator, test_generator = None, None, None
    if dataset == 'FB15K':
        train_path = os.path.join(path, "freebase_mtr100_mte100-train.txt")
        validation_path = os.path.join(path, "freebase_mtr100_mte100-valid.txt")
        test_path = os.path.join(path, "freebase_mtr100_mte100-test.txt")

        entity2id, relation2id = create_mappings(train_path)
        train_set = FB15KDataset(train_path, entity2id, relation2id)
        train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)
        validation_set = FB15KDataset(validation_path, entity2id, relation2id)
        validation_generator = torch_data.DataLoader(validation_set, batch_size=FLAGS.validation_batch_size)
        test_set = FB15KDataset(test_path, entity2id, relation2id)
        test_generator = torch_data.DataLoader(test_set, batch_size=FLAGS.validation_batch_size)
        print('FB15K data loaded')
    elif dataset == WORDNET:
        data_obj = pickle.load(open(os.path.join(path, 'wordnet_KG.pkl'), 'rb'))
        relation2id = data_obj['relation_idx_dict']
        entity2id = data_obj['word_idx_dict']
        train_set = WordNetDataset(data_obj['fact_list'])
        train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)

        print('WordNet data loaded')
    else:
        raise ValueError('')

    model = TransE(entity_count=len(entity2id), relation_count=len(relation2id), dim=vector_length, margin=margin,
                   device=device, norm=norm)  # type: torch.nn.Module
    model = model.to(device)
    optim_ = optim.SGD(model.parameters(), lr=learning_rate)
    summary_writer = tensorboard.SummaryWriter(log_dir=FLAGS.tensorboard_log_dir)
    start_epoch_id = 1
    step = 0
    best_score = 0.0

    if FLAGS.load_checkpoint_file is not None:
        checkpoint_path = os.path.join(FLAGS.checkpoint_folder, FLAGS.load_checkpoint_file)
        start_epoch_id, step, best_score = load_checkpoint(checkpoint_path, model, optim_)

    print(model)

    # Training loop
    for epoch_id in range(start_epoch_id, epochs + 1):
        print("Starting epoch: ", epoch_id)
        loss_impacting_samples_count = 0
        samples_count = 0
        model.train()

        for local_heads, local_relations, local_tails in train_generator:
            local_heads, local_relations, local_tails = (local_heads.to(device), local_relations.to(device),
                                                         local_tails.to(device))

            positive_triples = torch.stack((local_heads, local_relations, local_tails), dim=1)

            # Preparing negatives.
            # Generate binary tensor to replace either head or tail. 1 means replace head, 0 means replace tail.
            head_or_tail = torch.randint(high=2, size=local_heads.size(), device=device)
            random_entities = torch.randint(high=len(entity2id), size=local_heads.size(), device=device)
            broken_heads = torch.where(head_or_tail == 1, random_entities, local_heads)
            broken_tails = torch.where(head_or_tail == 0, random_entities, local_tails)
            negative_triples = torch.stack((broken_heads, local_relations, broken_tails), dim=1)

            optim_.zero_grad()

            loss, pd, nd = model(positive_triples, negative_triples)
            loss.mean().backward()

            summary_writer.add_scalar('Loss/train', loss.mean().data.cpu().numpy(), global_step=step)
            summary_writer.add_scalar('Distance/positive', pd.sum().data.cpu().numpy(), global_step=step)
            summary_writer.add_scalar('Distance/negative', nd.sum().data.cpu().numpy(), global_step=step)

            loss = loss.data.cpu()
            loss_impacting_samples_count += loss.nonzero().size()[0]
            samples_count += loss.size()[0]

            optim_.step()
            step += 1

        summary_writer.add_scalar('Metrics/loss_impacting_samples', loss_impacting_samples_count / samples_count * 100,
                                  global_step=epoch_id)
        print('epoch: {}, metrics/loss impacting samples: {}'
              .format(epoch_id, loss_impacting_samples_count/samples_count*100))

        if epoch_id > 0 and epoch_id % 200 == 0:
            save_checkpoint(model, optim_, epoch_id, step, best_score, FLAGS.checkpoint_folder, dataset)

        if epoch_id % FLAGS.validation_freq == 0:
            model.eval()
            if dataset == FB15K:
                _, _, hits_at_10, _ = test(model=model, data_generator=validation_generator,
                                           entities_count=len(entity2id),
                                           device=device, summary_writer=summary_writer,
                                           epoch_id=epoch_id, metric_suffix="val")

                score = hits_at_10
                if score > best_score:
                    best_score = score
                    save_checkpoint(model, optim_, epoch_id, step, best_score, FLAGS.checkpoint_folder, dataset)

    if dataset == FB15K:
        # Testing the best checkpoint on test dataset
        load_checkpoint(os.path.join(FLAGS.checkpoint_folder, FLAGS.load_checkpoint_file), model, optim_)
        best_model = model.to(device)
        best_model.eval()
        scores = test(model=best_model, data_generator=test_generator, entities_count=len(entity2id), device=device,
                      summary_writer=summary_writer, epoch_id=1, metric_suffix="test")
        print("Test scores: ", scores)


if __name__ == '__main__':
    app.run(main)
