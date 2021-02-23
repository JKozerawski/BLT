"""
# Copyright (c) Microsoft Corporation.
# Copyright (c) Jedrzej Kozerawski
# Licensed under the MIT license.

Partially based on https://github.com/zhmiao/OpenLongTailRecognition-OLTR
@author: Zhongqi Miao
Copyright (c) 2019, Zhongqi Miao
# Licensed under the BSD 3-Clause License.
"""
import os
import argparse
from data import dataloader
from run_networks import model
import warnings
from utils import source_import

# ================
# LOAD CONFIGURATIONS

data_root = {'ImageNet': '/data/jedrzej/ImageNet_LT',
             'Places': '/data/jedrzej/Places/',
             'iNaturalist_2018': '/media/scratch/jedrzej/workspace/data/iNaturalist_2018'}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/ImageNet_LT/stage_2.py', type=str)

parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--output_logits', default=False, action='store_true')

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--target', default=0.2, type=float)
parser.add_argument('--thresh', default=0.25, type=float)
parser.add_argument('--no_hallucinations', default=False, action='store_true')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--scale', default=16, type=int)
parser.add_argument('--margin', default=0.5, type=float)
parser.add_argument('--logit_margin', default=0.0, type=float)
parser.add_argument('--name', default="", type=str)
parser.add_argument('--backbone', default="res10", type=str)
parser.add_argument('--difficulty', default=1, type=int)
parser.add_argument('--save_final', default=False, action='store_true')
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--cb_loss', default=False, action='store_true')
parser.add_argument('--no_sampler', default=False, action='store_true')
parser.add_argument('--split_threshold', default=20, type=int)
parser.add_argument('--margin_loss', default="none", type=str)
parser.add_argument('--augmentations', default=False, action='store_true')
parser.add_argument('--hallucination_epoch', default=2, type=int)
args = parser.parse_args()

test_mode = args.test

output_logits = args.output_logits

config = source_import(args.config).config
training_opt = config['training_opt']
training_opt['batch_size'] = args.batch_size
# change
dataset = training_opt['dataset']


# possible parameters:
parameters = dict()
parameters["device"] = args.gpu
parameters["target_confidence"] = args.target
parameters["hallucination_threshold"] = args.thresh
parameters["no hallucinations"] = args.no_hallucinations
parameters["batch size"] = args.batch_size
parameters["scale"] = args.scale
parameters["margin"] = args.margin
parameters["logit margin"] = args.logit_margin
parameters["name"] = args.name
parameters["backbone"] = args.backbone
parameters["difficulty"] = args.difficulty
parameters["save final"] = args.save_final
parameters["split threshold"] = args.split_threshold
parameters["margin loss"] = args.margin_loss
parameters["hallucination_epoch"] = args.hallucination_epoch
parameters["data root"] = config['data_root']
parameters["sampler"] = training_opt['sampler']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])


if not test_mode:
    data = {x: dataloader.load_data(data_root=config['data_root'], dataset=dataset, phase=x,
                                    batch_size=args.batch_size,
                                    use_sampler=parameters["sampler"],
                                    num_workers=training_opt['num_workers'], gamma=args.gamma)
            for x in ['train', 'val', 'test']}

    training_model = model(config, data, parameters=parameters, test=False)
    training_model.train()
else:
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    print('Under testing phase, we load training data simply to calculate training data number for each class.')
    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                    batch_size=args.batch_size,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False)
            for x in ['train', 'test']}

    training_model = model(config, data, parameters=parameters, test=True)
    training_model.load_model()
    training_model.eval(phase='test')
    

print('ALL COMPLETED.')
