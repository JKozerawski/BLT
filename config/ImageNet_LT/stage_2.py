"""
# Copyright (c) Microsoft Corporation.
# Copyright (c) Jedrzej Kozerawski
# Licensed under the MIT license.

Partially based on https://github.com/zhmiao/OpenLongTailRecognition-OLTR
@author: Zhongqi Miao
Copyright (c) 2019, Zhongqi Miao
# Licensed under the BSD 3-Clause License.
"""

# Testing configurations
config = {}

config['data_root'] = '/data/jedrzej/ImageNet_LT'

training_opt = {}
training_opt['dataset'] = 'ImageNet_LT'
training_opt['log_dir'] = './logs/ImageNet_LT/stage2'
training_opt['num_classes'] = 1000
training_opt['batch_size'] = 256
training_opt['num_workers'] = 4
training_opt['num_epochs'] = 60
training_opt['display_step'] = 250
training_opt['feature_dim'] = 512
training_opt['open_threshold'] = 0.1
training_opt['sampler'] = True
training_opt['scheduler_params'] = {'step_size':20, 'gamma':0.1}
config['training_opt'] = training_opt

networks = {}
feature_param = {'use_modulatedatt': False, 'use_fc': True, 'dropout': None,
			     'stage1_weights':True, 'dataset': training_opt['dataset']}
feature_optim_param = {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0005}
networks['feat_model'] = {'def_file': './models/FeatureExtractor.py',
                          'params': feature_param,
                          'optim_params': feature_optim_param,
                          'fix': False}
classifier_param = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes'], 
				    'stage1_weights': True, 'dataset': training_opt['dataset']}
classifier_optim_param = {'lr': 0.1, 'momentum':0.9, 'weight_decay': 0.0005}
networks['classifier'] = {'def_file': './models/Classifier.py',
                          'params': classifier_param,
                          'optim_params': classifier_optim_param}
config['networks'] = networks

criterions = {}
perf_loss_param = {}
criterions['PerformanceLoss'] = {'def_file': './loss/SoftmaxLoss.py', 'loss_params': perf_loss_param,
                                 'optim_params': None, 'weight': 1.0}
config['criterions'] = criterions
