#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add the parent directory to the Python path

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
import gc

gc.collect()
torch.cuda.empty_cache()


def parse_args():
	parser = argparse.ArgumentParser(
		description='Training and Testing Knowledge Graph Embedding Models',
		usage='train.py [<args>] [-h | --help]'
	)

	#sampleing 参数
	parser.add_argument('--data_path', type=str, default='../data/drugbank/')
	parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)

	parser.add_argument('--default_subsampling', action='store_true',default=False)
	parser.add_argument('--uniq_based_subsampling', action='store_true',default=False)
	parser.add_argument('--freq_based_subsampling', action='store_true',default=False)
	parser.add_argument('--ignore_scoring_margin', action='store_true',default=False)


	# lte参数
	parser.add_argument('-lte', '--lte_operation', action='store_true', default=False)
	parser.add_argument('-id', '--input_dim', dest='input_dim', default=1000, type=int,
						help='Initial dimension size for entities and relations')
	parser.add_argument('-od', '--out_dim', dest='out_dim', default=1000,
						type=int, help='Number of out units in linear')
	parser.add_argument('-hd', '--hid_drop', dest='hid_drop',
						default=0.2, type=float, help='Dropout')
	parser.add_argument('--x_ops', dest='x_ops', default='p')
	parser.add_argument('--r_ops', dest='r_ops', default="")

	# kge参数
	parser.add_argument('--do_train', action='store_true')
	parser.add_argument('--do_valid', action='store_true')
	parser.add_argument('--do_test', action='store_true')
	parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

	parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
	parser.add_argument('--regions', type=int, nargs='+', default=None,
						help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

	parser.add_argument('--model', default='RotatE', type=str)
	parser.add_argument('-de', '--double_entity_embedding', action='store_true')
	parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

	parser.add_argument('-n', '--negative_sample_size', default=256, type=int)
	parser.add_argument('-d', '--hidden_dim', default=1000, type=int)
	parser.add_argument('-g', '--gamma', default=24.0, type=float)
	parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
	parser.add_argument('-sum', '--sum_ns_loss', action='store_true')
	parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
	parser.add_argument('-t', '--temperature', default=1.0, type=float)
	parser.add_argument('-b', '--batch_size', default=2048, type=int)
	parser.add_argument('-r', '--regularization', default=0.0, type=float)
	parser.add_argument('--test_batch_size', default=32, type=int, help='valid/test batch size')
	parser.add_argument('--uni_weight', action='store_true',
						help='Otherwise use subsampling weighting like in word2vec')

	parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
	parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
	parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
	parser.add_argument('-save', '--save_path', default='models/RotatE_Dataset1/', type=str)
	parser.add_argument('--max_steps', default=20000, type=int)
	parser.add_argument('--warm_up_steps', default=None, type=int)

	parser.add_argument('--save_checkpoint_steps', default=1000, type=int)
	parser.add_argument('--valid_steps', default=10000, type=int)
	parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
	parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
	parser.add_argument('--kmeans_step', default=100, type=int, help='')

	parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
	parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

	args = parser.parse_args()

	return args


args = parse_args()


def override_config(args):
	'''
	Override model and data configuration
	'''

	with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
		argparse_dict = json.load(fjson)

	args.countries = argparse_dict['countries']
	if args.data_path is None:
		args.data_path = argparse_dict['data_path']
	args.model = argparse_dict['model']
	args.double_entity_embedding = argparse_dict['double_entity_embedding']
	args.double_relation_embedding = argparse_dict['double_relation_embedding']
	args.hidden_dim = argparse_dict['hidden_dim']
	args.test_batch_size = argparse_dict['test_batch_size']


def save_kgemodel(model, optimizer, save_variable_list, args):
	'''
	Save the parameters of the model and the optimizer,
	as well as some other variables such as step and learning_rate
	'''

	argparse_dict = vars(args)
	with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
		json.dump(argparse_dict, fjson)

	torch.save({
		**save_variable_list,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict()},
		os.path.join(args.save_path, 'checkpoint')
	)

	entity_embedding = model.entity_embedding.detach().cpu().numpy()

	np.save(
		os.path.join(args.save_path, 'entity_embedding'),
		entity_embedding
	)

	relation_embedding = model.relation_embedding.detach().cpu().numpy()

	np.save(
		os.path.join(args.save_path, 'relation_embedding'),
		relation_embedding
	)


def read_triple(file_path, entity2id, relation2id):
	'''
	Read triples and map them into ids.
	'''
	triples = []
	with open(file_path) as fin:
		for line in fin:
			h, r, t = line.strip('\n').split('\t')
			triples.append((entity2id[h], relation2id[r], entity2id[t]))
	return triples

def read_drugbank_triple(file_path, entity2id, relation2id):
	'''
	Read triples and map them into ids.
	'''
	triples = []
	with open(file_path) as fin:
		for line in fin:
			h, t, r = line.strip().split()
			triples.append((entity2id[h], relation2id[r], entity2id[t]))
	return triples

def read_snap_triple(file_path, entity2id, relation2id):
	'''
	Read triples and map them into ids.
	'''
	triples = []
	with open(file_path) as fin:
		for line in fin:
			h, r, t = line.strip().split()
			triples.append((entity2id[h], relation2id[r], entity2id[t]))
	return triples


def set_logger(args):
	'''
	Write logs to checkpoint and console
	'''

	if args.do_train:
		log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
	else:
		log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

	logging.basicConfig(
		format='%(asctime)s %(levelname)-8s %(message)s',
		level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S',
		filename=log_file,
		filemode='a'
	)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
	'''
	Print the evaluation logs
	'''
	for metric in metrics:
		logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

CUDA = torch.cuda.is_available()


def train_kge(args):

	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	if (not args.do_train) and (not args.do_valid) and (not args.do_test):
		raise ValueError('one of train/val/test mode must be choosed.')

	if args.init_checkpoint:
		override_config(args)
	elif args.data_path is None:
		raise ValueError('one of init_checkpoint/data_path must be choosed.')

	if args.do_train and args.save_path is None:
		raise ValueError('Where do you want to save your trained model?')

	if args.save_path and not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	# Write logs to checkpoint and console
	set_logger(args)

	with open(os.path.join(args.data_path, 'entity2id.txt'), encoding='utf-8') as fin:
		entity2id = dict()
		for line in fin:
			entity, eid = line.strip('\n').split('\t')
			entity2id[entity] = int(eid)

	with open(os.path.join(args.data_path, 'relation2id.txt'), encoding='utf-8') as fin:
		relation2id = dict()
		for line in fin:
			relation, rid = line.strip('\n').split('\t')
			relation2id[relation] = int(rid)

	nentity = len(entity2id)
	nrelation = len(relation2id)

	args.nentity = nentity
	args.nrelation = nrelation

	logging.info('kge_Model: %s' % args.model)
	logging.info('Data Path: %s' % args.data_path)
	logging.info('#entity: %d' % nentity)
	logging.info('#relation: %d' % nrelation)

	if 'drugbank' in args.data_path:
		train_triples = read_drugbank_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
		logging.info('#train: %d' % len(train_triples))
		valid_triples = read_drugbank_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
		logging.info('#valid: %d' % len(valid_triples))
		test_triples = read_drugbank_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
		logging.info('#test: %d' % len(test_triples))
	else:
		train_triples = read_snap_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
		logging.info('#train: %d' % len(train_triples))
		valid_triples = read_snap_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
		logging.info('#valid: %d' % len(valid_triples))
		test_triples = read_snap_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
		logging.info('#test: %d' % len(test_triples))

	logging.info('#train: %d' % len(train_triples))
	logging.info('#valid: %d' % len(valid_triples))
	logging.info('#test: %d' % len(test_triples))

	# All true triples
	all_true_triples = train_triples + valid_triples + test_triples

	kge_model = KGEModel(
		model_name=args.model,
		nentity=nentity,
		nrelation=nrelation,
		hidden_dim=args.hidden_dim,
		gamma=args.gamma,
		double_entity_embedding=args.double_entity_embedding,
		double_relation_embedding=args.double_relation_embedding,
		ignore_scoring_margin=args.ignore_scoring_margin,
		lte_operation=args.lte_operation,
		input_dim=args.input_dim,
		out_dim=args.out_dim,
		hid_drop=args.hid_drop,
		x_ops=args.x_ops,
		r_ops=args.r_ops,
	)

	optimizer_params = {'lr': 0.001, 'weight_decay': 1e-5}

	logging.info('Model Parameter Configuration:')
	for name, param in kge_model.named_parameters():
		logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

	if args.cuda:
		kge_model = kge_model.cuda()

	if args.do_train:
		# Set training dataloader iterator
		train_dataloader_head = DataLoader(
			TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
			batch_size=args.batch_size,
			shuffle=True,
			num_workers=max(1, args.cpu_num // 2),
			collate_fn=TrainDataset.collate_fn
		)

		train_dataloader_tail = DataLoader(
			TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
			batch_size=args.batch_size,
			shuffle=True,
			num_workers=max(1, args.cpu_num // 2),
			collate_fn=TrainDataset.collate_fn
		)

		train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

		# Set training configuration
		current_learning_rate = args.learning_rate
		optimizer = torch.optim.Adam(
			filter(lambda p: p.requires_grad, kge_model.parameters()),
			lr=current_learning_rate
		)
		if args.warm_up_steps:
			warm_up_steps = args.warm_up_steps
		else:
			warm_up_steps = args.max_steps // 2

	if args.init_checkpoint:
		# Restore model from checkpoint directory
		logging.info('Loading checkpoint %s...' % args.init_checkpoint)
		checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
		init_step = checkpoint['step']
		kge_model.load_state_dict(checkpoint['model_state_dict'])
		if args.do_train:
			current_learning_rate = checkpoint['current_learning_rate']
			warm_up_steps = checkpoint['warm_up_steps']
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	else:
		logging.info('Ramdomly Initializing %s Model...' % args.model)
		init_step = 0

	step = init_step

	logging.info('Start Training KGE...')
	logging.info('init_step = %d' % init_step)
	logging.info('batch_size = %d' % args.batch_size)
	logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
	logging.info('hidden_dim = %d' % args.hidden_dim)
	logging.info('gamma = %f' % args.gamma)
	logging.info('negative_sampling_size = %d' % args.negative_sample_size)
	if args.negative_adversarial_sampling:
		logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

	# Set valid dataloader as it would be evaluated during training

	if args.do_train:
		logging.info('learning_rate = %d' % current_learning_rate)

		training_logs = []

		for step in range(init_step, args.max_steps):

			log = kge_model.train_step(kge_model, optimizer, train_iterator, args)

			training_logs.append(log)

			# if step >= warm_up_steps:
			if step % 10000 == 0:
				current_learning_rate = current_learning_rate / 10
				logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
				optimizer = torch.optim.Adam(
					filter(lambda p: p.requires_grad, kge_model.parameters()),
					lr=current_learning_rate
				)
				warm_up_steps = warm_up_steps * 3

			if step % args.save_checkpoint_steps == 0:
				save_variable_list = {
					'step': step,
					'current_learning_rate': current_learning_rate,
					'warm_up_steps': warm_up_steps
				}
				save_kgemodel(kge_model, optimizer, save_variable_list, args)

			if step % args.log_steps == 0:
				metrics = {}
				for metric in training_logs[0].keys():
					metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
				log_metrics('Training average', step, metrics)
				training_logs = []

			if args.do_valid and step % args.valid_steps == 0:
				logging.info('Evaluating on Valid Dataset...')
				metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
				log_metrics('Valid', step, metrics)

		save_variable_list = {
			'step': step,
			'current_learning_rate': current_learning_rate,
			'warm_up_steps': warm_up_steps
		}
		save_kgemodel(kge_model, optimizer, save_variable_list, args)

	if args.do_valid:
		logging.info('Evaluating on Valid Dataset...')
		metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
		log_metrics('Valid', step, metrics)

	if args.do_test:
		logging.info('Evaluating on Test Dataset...')
		metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
		log_metrics('Test', step, metrics)
		if "drugbank" in args.data_path:
			kge_model.test_drugbank_PSE(kge_model, test_triples, all_true_triples, entity2id, relation2id, args)
		else:
			kge_model.test_PSE(kge_model, test_triples, all_true_triples, entity2id, relation2id, args)

	if args.evaluate_train:
		logging.info('Evaluating on Training Dataset...')
		metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
		log_metrics('Train', step, metrics)


if __name__ == '__main__':
	args = parse_args()
	# logging.info(args)
	print(args)
	train_kge(args)
