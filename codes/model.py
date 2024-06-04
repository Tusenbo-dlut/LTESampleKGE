#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from metrics import average_precision, APN, precision_at_k
from metrics.classification import auc_pr, auc_roc

torch.autograd.set_detect_anomaly(True)

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param

CUDA = torch.cuda.is_available()  # checking cuda availability

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, input_dim, out_dim, hid_drop,x_ops,r_ops,
                 double_entity_embedding=False, double_relation_embedding=False, lte_operation = False, ignore_scoring_margin=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hid_drop = hid_drop
        self.epsilon = 2.0
        self.x_ops = x_ops
        self.r_ops = r_ops

        if ignore_scoring_margin:
            self.gamma = nn.Parameter(
                torch.Tensor([0.0]),
                requires_grad=False
            )


        #lte参数
        self.lte_operation = lte_operation
        self.loop_emb = get_param([1, self.input_dim])
        self.h_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.input_dim, self.out_dim, bias=False),
            'b': nn.BatchNorm1d(self.out_dim),
            'd': nn.Dropout(self.hid_drop),
            'a': nn.Tanh(),
        })

        self.t_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.input_dim, self.out_dim, bias=False),
            'b': nn.BatchNorm1d(self.out_dim),
            'd': nn.Dropout(self.hid_drop),
            'a': nn.Tanh(),
        })

        self.r_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.input_dim//2, self.out_dim//2, bias=False),
            'b': nn.BatchNorm1d(self.out_dim),
            'd': nn.Dropout(self.hid_drop),
            'a': nn.Tanh(),
        })

        self.x_ops = self.x_ops
        self.r_ops = self.r_ops
        self.diff_ht = False

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Multi-head attention mechanism

        if self.lte_operation :
          logging.info('init vector matrices join linear transformations')
        else:
          logging.info('init vector matrices not join linear transformations')

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')


    def exop(self, x, r, x_ops=None, r_ops=None, diff_ht=False):
        # x:所有实体的初始化向量
        x_head = x_tail = x
        if len(x_ops) > 0:
            for x_op in x_ops.split("."):
                if diff_ht:
                    x_head = self.h_ops_dict[x_op](x_head)
                    x_tail = self.t_ops_dict[x_op](x_tail)
                else:
                    x_head = x_tail = self.h_ops_dict[x_op](x_head)

        if len(r_ops) > 0:
            for r_op in r_ops.split("."):
                r = self.r_ops_dict[r_op](r)

        return x_head, x_tail, r

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        #加入线性变换后的h,t,r向量矩阵
        if self.lte_operation :
            # x_h, x_t, r = self.exop(self.entity_embedding - self.loop_emb, self.relation_embedding, self.x_ops, self.r_ops)
            x_h, x_t, r = self.exop(self.entity_embedding, self.relation_embedding, self.x_ops,
                                    self.r_ops)
            # print('init vector matrices join linear transformations\n')
            # print('entity matrices dim:',x_h.shape)
            # print('relation matrices dim:', r.shape)
        # 不加入线性变换
        else:
            x_h = self.entity_embedding
            x_t = self.entity_embedding
            r = self.relation_embedding
            # print('init vector matrices not join linear transformations\n')
            # print('entity matrices dim:', x_h.shape)
            # print('relation matrices dim:', r.shape)


        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                x_h,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                r,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                x_t,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                x_h,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                r,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                x_t,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                x_h,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                r,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                x_t,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        # positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        positive_sample, negative_sample, hr_freq, tr_freq, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            # subsampling_weight = subsampling_weight.cuda()
            hr_freq = hr_freq.cuda()
            tr_freq = tr_freq.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        positive_score = model(positive_sample)

        batch_size = positive_score.shape[0]

        if args.freq_based_subsampling:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)
            else:
                if args.sum_ns_loss:
                    negative_score = F.logsigmoid(-negative_score).sum(dim=1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim=1)
            if mode == 'head-batch':
                subsampling_weight = tr_freq
            if mode == 'tail-batch':
                subsampling_weight = hr_freq
            conditional_weight = (hr_freq + tr_freq).cuda()
            conditional_weight = torch.sqrt(1 / conditional_weight)
            subsampling_weight = torch.sqrt(1 / subsampling_weight)
            subsampling_weight = subsampling_weight.cuda()
            positive_sample_loss = - (conditional_weight * positive_score).sum() / conditional_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        elif args.uniq_based_subsampling:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)
            else:
                if args.sum_ns_loss:
                    negative_score = F.logsigmoid(-negative_score).sum(dim=1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim=1)
            if mode == 'head-batch':
                subsampling_weight = tr_freq
            if mode == 'tail-batch':
                subsampling_weight = hr_freq
            subsampling_weight = torch.sqrt(1 / subsampling_weight)
            subsampling_weight = subsampling_weight.cuda()
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        elif args.default_subsampling:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)
            else:
                if args.sum_ns_loss:
                    negative_score = F.logsigmoid(-negative_score).sum(dim=1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim=1)
            subsampling_weight = hr_freq + tr_freq
            subsampling_weight = torch.sqrt(1 / subsampling_weight)
            subsampling_weight = subsampling_weight.cuda()
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        else:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)
            else:
                if args.sum_ns_loss:
                    negative_score = F.logsigmoid(-negative_score).sum(dim=1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim=1)
            positive_sample_loss = - (positive_score).sum() / batch_size
            negative_sample_loss = - (negative_score).sum() / batch_size

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()  # 测试模式

        if args.countries:  # 针对论文中的country数据集
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,  # 所有的正确的三元组
                    args.nentity,  # 所有的实体个数
                    args.nrelation,  # 所有的关系个数
                    'head-batch'
                ),
                batch_size=args.test_batch_size,  # 16
                num_workers=max(1, args.cpu_num // 2),  # 5
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])  # head-batch和tail-batch的样本数
            # train时:  positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias  # 选择出来的三元组(head,relation,tail1),其他的(head,relation,tail2)也会分数比较高，对这些分数减去1,降低他们的排名，使其不会干扰对性能的判断

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1,
                                                descending=True)  # 分数越大越好 返回从大到小排序后的值所对应原a的下标，即torch.sort()返回的indices
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]  # 正确的头实体的编号
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]  # 正确的尾实体的编号
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[
                                i]).nonzero()  # nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
                            assert ranking.size(0) == 1  # 如果ranking.size(0) == 1，程序正常往下运行

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            # 指标
                            logs.append({
                                'MRR': 1.0 / ranking,  # 越大越好
                                'MR': float(ranking),  # MRR倒数
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)  # 对log的每一个metric都求平均值

        return metrics

    @staticmethod
    def test_PSE(model, test_triples, all_true_triples, entity2id, relation2id, args):

        model.eval()

        entity_set = [value for value in entity2id.values()]
        relation_set = [value for value in relation2id.values()]

        # file = open(args.data_path + '/PES.txt', 'a')
        logging.info(args.model, args.lte_operation, args.default_subsampling, args.freq_based_subsampling, args.freq_based_subsampling, args.max_steps, args.learning_rate, args.hidden_dim, args.gamma)
        # print("model: {} - lte_operation: {} - x_ops: {} - r_ops: {} - default_subsampling: {} - freq_based_subsampling: {} - uniq_based_subsampling: {} - max_steps {} - learning_rate: {} - hidden_dim: {} - gamma: {}" .format
        #       (args.model, args.lte_operation, args.x_ops, args.r_ops, args.default_subsampling, args.freq_based_subsampling, args.uniq_based_subsampling, args.max_steps, args.learning_rate, args.hidden_dim, args.gamma), flush=True, file=file)
        metrics_per_se = {se_idx: {"ap": .0, "auc-roc": .0, "auc-pr": .0, "AP@n": .0} for se_idx in relation_set}

        se_ap_list = []
        se_auc_roc_list = []
        se_auc_pr_list = []
        se_p50_list = []

        drug_combinations = np.array(
            [[d1, d2] for d1, d2 in list(itertools.product(entity_set, entity_set)) if d1 != d2])

        se_facts_full_dict = {se: set() for se in relation_set}
        bench_idx_data = all_true_triples
        for s, p, o in bench_idx_data:
            se_facts_full_dict[p].add((s, p, o))

        print("================================================================================")
        for se in tqdm(relation_set, desc="Evaluating test data for each side-effect"):
            se_code = len(se_facts_full_dict[se])
            se_all_facts_set = se_facts_full_dict[se]
            se_test_facts_pos = np.array([[s, p, o] for s, p, o in test_triples if p == se])
            se_test_facts_pos_size = len(se_test_facts_pos)

            se_test_facts_neg = np.array([[d1, se, d2] for d1, d2 in drug_combinations
                                          if (d1, se, d2) not in se_all_facts_set
                                          and (d2, se, d1) not in se_all_facts_set])

            # shuffle and keep negatives with size equal to positive instances so positive to negative ratio is 1:1 （一对一采样）
            np.random.shuffle(se_test_facts_neg)
            se_test_facts_neg = se_test_facts_neg[:se_test_facts_pos_size, :]

            set_test_facts_all = np.concatenate([se_test_facts_pos, se_test_facts_neg])
            se_test_facts_labels = np.concatenate(
                [np.ones([len(se_test_facts_pos)]), np.zeros([len(se_test_facts_neg)])])
            set_test_facts_all = torch.LongTensor(set_test_facts_all)
            set_test_facts_all = set_test_facts_all.cuda()
            se_test_facts_scores = model(set_test_facts_all).squeeze(1).cpu().detach().numpy()

            # se_test_facts_scores = minmax_scale(se_test_facts_scores) / np.sum(se_test_facts_scores)

            se_ap = average_precision(se_test_facts_labels, se_test_facts_scores)
            se_p50 = precision_at_k(se_test_facts_labels, se_test_facts_scores, k=50)
            se_auc_pr = auc_pr(se_test_facts_labels, se_test_facts_scores)
            se_auc_roc = auc_roc(se_test_facts_labels, se_test_facts_scores)

            se_ap_list.append(se_ap)
            se_auc_roc_list.append(se_auc_roc)
            se_auc_pr_list.append(se_auc_pr)
            se_p50_list.append(se_p50)

            metrics_per_se[se] = {"ap": se_ap, "auc-roc": se_auc_roc, "auc-pr": se_auc_pr, "AP@50": se_p50}
            logging.info("AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - AP@50: %1.4f ---------> %s" %
                  (se_ap, se_auc_roc, se_auc_pr,se_p50, se_code))
            # print("AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - AP@50: %1.4f ---------> %s" %
            #       (se_ap, se_auc_roc, se_auc_pr,se_p50, se_code), flush=True, file=file)


        se_ap_list_avg = np.average(se_ap_list)
        se_auc_roc_list_avg = np.average(se_auc_roc_list)
        se_auc_pr_list_avg = np.average(se_auc_pr_list)
        se_p50_list_avg = np.average(se_p50_list)

        # print("================================================================================", file=file)
        logging.info("[AVERAGE] AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - AP@50: %1.4f" %
              (se_ap_list_avg, se_auc_roc_list_avg, se_auc_pr_list_avg, se_p50_list_avg))
        # print("[AVERAGE] AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - AP@50: %1.4f" %
        #       (se_ap_list_avg, se_auc_roc_list_avg, se_auc_pr_list_avg, se_p50_list_avg), flush=True, file=file)
        # print("================================================================================", file=file)

    @staticmethod
    def test_drugbank_PSE(model, test_triples, all_true_triples, entity2id, relation2id, args):

        model.eval()

        entity_set = [value for value in entity2id.values()]
        relation_set = [value for value in relation2id.values()]

        # file = open(args.data_path + '/PES.txt', 'a')
        logging.info(args.model, args.lte_operation, args.default_subsampling, args.freq_based_subsampling,
                     args.freq_based_subsampling, args.max_steps, args.learning_rate, args.hidden_dim, args.gamma)

        metrics_per_se = {se_idx: {"ap": .0, "auc-roc": .0, "auc-pr": .0, "AP@n": .0} for se_idx in relation_set}

        se_ap_list = []
        se_auc_roc_list = []
        se_auc_pr_list = []
        se_pn_list = []

        drug_combinations = np.array(
            [[d1, d2] for d1, d2 in list(itertools.product(entity_set, entity_set)) if d1 != d2])

        se_facts_full_dict = {se: set() for se in relation_set}
        bench_idx_data = all_true_triples
        for s, p, o in bench_idx_data:
            se_facts_full_dict[p].add((s, p, o))

        print("================================================================================")
        for se in tqdm(relation_set, desc="Evaluating test data for each side-effect"):
            se_code = len(se_facts_full_dict[se])
            se_all_facts_set = se_facts_full_dict[se]
            se_test_facts_pos = np.array([[s, p, o] for s, p, o in test_triples if p == se])
            se_test_facts_pos_size = len(se_test_facts_pos)

            se_test_facts_neg = np.array([[d1, se, d2] for d1, d2 in drug_combinations
                                          if (d1, se, d2) not in se_all_facts_set
                                          and (d2, se, d1) not in se_all_facts_set])

            # shuffle and keep negatives with size equal to positive instances so positive to negative ratio is 1:1 （一对一采样）
            np.random.shuffle(se_test_facts_neg)
            se_test_facts_neg = se_test_facts_neg[:se_test_facts_pos_size, :]

            set_test_facts_all = np.concatenate([se_test_facts_pos, se_test_facts_neg])
            se_test_facts_labels = np.concatenate(
                [np.ones([len(se_test_facts_pos)]), np.zeros([len(se_test_facts_neg)])])
            set_test_facts_all = torch.LongTensor(set_test_facts_all)
            set_test_facts_all = set_test_facts_all.cuda()
            se_test_facts_scores = model(set_test_facts_all).squeeze(1).cpu().detach().numpy()

            # se_test_facts_scores = minmax_scale(se_test_facts_scores) / np.sum(se_test_facts_scores)

            se_ap = average_precision(se_test_facts_labels, se_test_facts_scores)
            se_pn = APN(se_test_facts_labels, se_test_facts_scores)
            se_auc_pr = auc_pr(se_test_facts_labels, se_test_facts_scores)
            se_auc_roc = auc_roc(se_test_facts_labels, se_test_facts_scores)

            se_ap_list.append(se_ap)
            se_auc_roc_list.append(se_auc_roc)
            se_auc_pr_list.append(se_auc_pr)
            se_pn_list.append(se_pn)


            metrics_per_se[se] = {"ap": se_ap, "auc-roc": se_auc_roc, "auc-pr": se_auc_pr, "AP@n": se_pn}
            logging.info("AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - AP@n: %1.4f ---------> %s" %
                         (se_ap, se_auc_roc, se_auc_pr, se_pn, se_code))
            # print("AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - AP@n: %1.4f ---------> %s" %
            #       (se_ap, se_auc_roc, se_auc_pr, se_pn, se_code), flush=True, file=file)

        se_ap_list_avg = np.average(se_ap_list)
        se_auc_roc_list_avg = np.average(se_auc_roc_list)
        se_auc_pr_list_avg = np.average(se_auc_pr_list)
        se_pn_list_avg = np.average(se_pn_list)

        # print("================================================================================", file=file)
        logging.info("[AVERAGE] AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - AP@n: %1.4f" %
                     (se_ap_list_avg, se_auc_roc_list_avg, se_auc_pr_list_avg, se_pn_list_avg))
        # print("[AVERAGE] AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - AP@n: %1.4f" %
        #       (se_ap_list_avg, se_auc_roc_list_avg, se_auc_pr_list_avg, se_pn_list_avg), flush=True,
        #       file=file)
        # print("================================================================================", file=file)
