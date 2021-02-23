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
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import time
import numpy as np
from gradient_ascent_adv import DisguisedFoolingSampleGeneration
from PIL import Image


class model():
    
    def __init__(self, config, data, parameters, test=False):

        # Default Parameters:
        self.dev_id = int(parameters["device"])
        self.epoch_to_hallucinate_on_the_fly = parameters["hallucination_epoch"]
        self.target_confidence = parameters["target_confidence"]
        self.hallucination_threshold = parameters["hallucination_threshold"]
        self.no_hallucinations = parameters["no hallucinations"]
        self.scale = parameters["scale"]
        self.margin = parameters["margin"]
        self.logit_margin = parameters["logit margin"]
        self.save_final = parameters["save final"]
        self.difficulty = parameters["difficulty"]
        self.name = parameters["name"]
        self.backbone = parameters["backbone"]
        self.data_root = parameters["data root"]
        self.split_threshold = parameters["split threshold"]
        self.batch_size = parameters["batch size"]
        self.training_file = "./data/" + config['training_opt']['dataset'] + "/" + config['training_opt']['dataset'] + "_train.txt"
        self.end_epoch = 55
        self.epoch = 1
        self.milestones = [23, 38, 52]


        if 'stage1' in config['training_opt']['log_dir']:
            self.end_epoch = 35
            self.milestones = [25, 32]
        else:
            self.end_epoch = 55
            self.milestones = [23, 38, 52]

        if self.no_hallucinations:
            self.epoch_to_hallucinate_on_the_fly = self.end_epoch + 2

        device = 'cuda:' + str(self.dev_id)

        print("PARAMETERS:")
        print("Using device:", device)
        print("Setting target confidence to:", self.target_confidence)
        print("Hallucination threshold:", self.hallucination_threshold)
        print("Using hallucination:", not self.no_hallucinations)
        print("Batch size:", self.batch_size)
        print("Scale:", self.scale)
        print("Margin:", self.margin)
        print("Logit margin:", self.logit_margin)
        print("Name to save:", self.name)
        print("Backbone used:", self.backbone)
        print("Difficulty of new samples:", self.difficulty)
        print("Saving final model instead of best:", self.save_final)
        print("Splitting branches on threshold:", self.split_threshold)
        print("Log directory:",  config['training_opt']['log_dir'])
        self.device_name = device
        torch.cuda.set_device(device)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')


        self.config = config
        self.training_opt = self.config['training_opt']
        self.data = data
        self.test_mode = test


        # get few-shot categories:
        with open(self.training_file, 'rb') as txtfile:
            lines = txtfile.readlines()
        self.label_count = np.zeros(self.training_opt['num_classes'])
        self.paths = []
        self.labels = []
        self.image_paths = []
        self.image_labels = []
        for line in lines:
            lbl = int(line.split()[-1])
            self.image_paths.append(line.split()[0].decode("utf-8"))
            self.image_labels.append(lbl)
            self.label_count[lbl] += 1
        self.image_labels = np.asarray(self.image_labels)
        self.few_shot_labels = np.where(self.label_count <= 20)[0]
        self.medium_shot_labels = np.where(self.label_count > 20)[0]
        self.many_shot_labels = np.where(self.label_count > 100)[0]
        self.medium_shot_labels = np.asarray([lbl for lbl in self.medium_shot_labels if lbl not in self.many_shot_labels])
        print("Few shot categories:", len(self.few_shot_labels))
        print("Medium shot categories:", len(self.medium_shot_labels))
        print("Many shot categories:", len(self.many_shot_labels))

        self.small_categories = np.where(self.label_count <= self.split_threshold)[0]
        self.big_categories = np.where(self.label_count > self.split_threshold)[0]
        print("Found:", len(self.small_categories), "small categories.")
        print("Found:", len(self.big_categories), "big categories.")
        self.label_groups = [self.small_categories, self.big_categories]


        # Initialize model
        self.init_models()


        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps 
            # for each epoch based on actual number of training data instead of 
            # oversampled data number 
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num / self.training_opt['batch_size'])
            print("Dataset length:", self.training_data_num)
            print("Num of iterations per epoch:", self.epoch_steps)

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            self.init_criterions()

            
        # Set up log file
        self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')

        try:
            self.networks['classifier'].module.cosnorm_classifier.scale = self.scale
            self.networks['classifier'].module.cosnorm_classifier.margin = self.margin
        except:
            print("Cannot assign margin and scale")
        try:
            self.networks['classifier'].module.margin = self.logit_margin
        except:
            print("Cannot assign logit margin")

    def init_models(self):
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []
        for key, val in networks_defs.items():
            # Networks
            def_file = val['def_file']
            model_args = list(val['params'].values())
            model_args.append(self.test_mode)
            print()
            print(key, val)
            print(val['params'])
            print(self.backbone)
            if 'in_dim' in val['params']:
                self.networks[key] = source_import(def_file).create_model(*model_args)
            elif 'caffe' in val['params']:
                self.networks[key] = source_import(def_file).create_model(stage1_weights=val['params']['stage1_weights'],
                                                                      dataset=val['params']['dataset'], caffe=val['params']['caffe'],
                                                                      test=self.test_mode, backbone=self.backbone, device=self.device_name)

            elif 'stage1_weights' in val['params']:
                self.networks[key] = source_import(def_file).create_model(
                    stage1_weights=val['params']['stage1_weights'],
                    dataset=val['params']['dataset'],
                    test=self.test_mode, backbone=self.backbone, device=self.device_name)

            self.networks[key] = nn.DataParallel(self.networks[key], device_ids=[self.dev_id]).to(self.device)

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']})


    def init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = val['loss_params'].values()
            self.criterions[key] = source_import(def_file).create_loss(*loss_args).to(self.device)
            self.criterion_weights[key] = val['weight']
            print(key, val)
          
            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(), 'lr': optim_params['lr'], 'momentum': optim_params['momentum'], 'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler

                self.criterion_optimizer, self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler

    def batch_forward(self, inputs, labels=None, phase='train'):
        '''
        This is a general single batch running function. 
        '''
        # Calculate Features
        self.features = self.networks['feat_model'](inputs)

        if phase != 'train':
            use_labels = False
        else:
            use_labels = True

        if 'stage1' in self.config['training_opt']['log_dir']:
            self.logits, self.other = self.networks['classifier'](self.features)
        else:
            self.logits, self.other = self.networks['classifier'](self.features, labels, use_labels)

    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):
        self.loss_perf = self.criterions['PerformanceLoss'](self.other, labels) * self.criterion_weights['PerformanceLoss']
        self.loss = self.loss_perf



    def train(self):

        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_acc = 0.0
        best_epoch = 0

        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())


        # Loop over epochs
        for epoch in range(1, self.end_epoch + 1):
            self.epoch = epoch
            start_time = time.time()
            for model in self.networks.values():
                model.train()

            torch.cuda.empty_cache()
            
            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train() 
            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # Iterate over dataset

            for step, (inputs, labels, paths) in enumerate(self.data['train']):
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break

                if self.epoch >= self.epoch_to_hallucinate_on_the_fly:

                    # choose labels in a single pass:
                    np_labels = labels.numpy()
                    np_labels_bool = np.isin(np_labels, self.small_categories)
                    labels_to_transform = np.where(np_labels_bool==True)[0]
                    labels_to_transform = [np_labels[idx] for idx in labels_to_transform]
                    labels_to_transform = np.random.choice(labels_to_transform, int(self.hallucination_threshold*len(labels_to_transform)), replace=False)

                    new_images, new_labels = self.generate_adversarial_images(labels_to_transform)

                    # if there are any new images:
                    if len(new_labels) > 0:
                        new_images = torch.stack(new_images).cpu()
                        new_labels = torch.tensor(np.asarray(new_labels))
                        inputs = torch.cat([inputs, new_images], dim=0)
                        labels = torch.cat([labels, new_labels], dim=0)

                    del new_images, new_labels, np_labels_bool, np_labels, labels_to_transform
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # If training, forward with loss, and no top 5 accuracy calculation
                self.batch_forward(inputs, labels, phase='train')

                self.batch_loss(labels)
                self.batch_backward()

                # Output minibatch training results
                if step % self.training_opt['display_step'] == 0:
                    minibatch_loss_perf = self.loss_perf.item()
                    _, preds = torch.max(self.logits, 1)

                    minibatch_acc = mic_acc_cal(preds, labels)

                    print_str = ['Epoch: [%d/%d]' % (epoch, self.end_epoch),
                                 'Step: %5d' % (step),
                                 'Minibatch_loss_performance: %.3f' % (minibatch_loss_perf),
                                 'Minibatch_accuracy_micro: %.3f' % (minibatch_acc),
                                 'Time elapsed: %.2f' % (time.time()-start_time)]
                    print_write(print_str, self.log_file)
                    start_time = time.time()

            # After every epoch, validation
            self.eval(phase='val')
            # Under validation, the best model need to be updated
            # save the best model so far (best on val accuracy)
            if self.eval_acc_mic_top1+self.eval_acc_mic_top5 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1 #+self.eval_acc_mic_top5
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
                # Save the best model and best centroids if calculated
                self.save_model(epoch, best_epoch, best_model_weights, best_acc, model_name=self.name)
            if epoch % 2 == 0:
                self.eval(phase='test')
        print()
        print('Training Complete.')
        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        print('Done')

    def eval(self, phase='val', openset=False):

        self.curr_confusion_matrix = np.zeros((self.training_opt['num_classes'], self.training_opt['num_classes']))
        print_str = ['Phase: %s' % phase]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval()
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        topk = (1, 5)
        maxk = max(topk)

        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, phase=phase)
                self.total_labels = torch.cat((self.total_labels, labels))
                self.total_logits = torch.cat((self.total_logits, self.logits))

        probsT = self.total_logits.detach()
        probs, preds = F.softmax(probsT, dim=1).max(dim=1)
        _, pred2 = F.softmax(probsT, dim=1).topk(maxk, 1, True, True)

        pred2 = pred2.t()
        correct = pred2.eq(self.total_labels.view(1, -1).expand_as(pred2))

        # Calculate top1 and top5 accuracy:
        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True).cpu().numpy()[0]
        self.eval_acc_mic_top1 = 100*correct_k/self.total_labels.size(0)
        correct_k = correct[:5].view(-1).float().sum(0, keepdim=True).cpu().numpy()[0]
        self.eval_acc_mic_top5 = 100*correct_k / self.total_labels.size(0)

        # Calculate the confusion matrix:
        pred = preds.cpu().numpy()
        lbls = self.total_labels.cpu().numpy()
        for i in range(len(pred)):
            self.curr_confusion_matrix[pred[i], lbls[i]] += 1
        self.confusion_matrix = self.curr_confusion_matrix

        # Calculate the per-group accuracy
        label_groups = [self.many_shot_labels, self.medium_shot_labels, self.few_shot_labels]
        self.many_acc_top1, self.median_acc_top1, self.low_acc_top1, self.many_acc_top5, self.median_acc_top5, self.low_acc_top5 = shot_acc(correct, self.total_labels, label_groups=label_groups, counts=self.label_count)

        # Top-1 accuracy and additional string
        print_str = ['Phase: %s' % phase, '\n',
                     'Evaluation_accuracy_micro_top1: %.3f' % self.eval_acc_mic_top1,
                     'Evaluation_accuracy_micro_top5: %.3f' % self.eval_acc_mic_top5, '\n',
                     'Many_shot_accuracy_top1: %.3f' % self.many_acc_top1,
                     'Median_shot_accuracy_top1: %.3f' % self.median_acc_top1,
                     'Low_shot_accuracy_top1: %.3f' % self.low_acc_top1, '\n',
                     'Many_shot_accuracy_top5: %.3f' % self.many_acc_top5,
                     'Median_shot_accuracy_top5: %.3f' % self.median_acc_top5,
                     'Low_shot_accuracy_top5: %.3f' % self.low_acc_top5, '\n']

        if phase == 'val':
            print_write(print_str, self.log_file)
        else:
            print(*print_str)
        del self.total_labels, self.total_logits, probsT, probs, preds, pred, pred2, lbls

    def generate_adversarial_images(self, labels):
        train_set = self.data_root+"/"

        images = []
        targets = []
        new_labels = []
        for i in range(len(labels)):
            # get the target:
            indices = np.where(self.confusion_matrix[..., labels[i]] > 0)[0]
            indices = [idx for idx in indices if idx != labels[i]]
            if len(indices) > 0:
                weights = np.asarray([self.confusion_matrix[idx, labels[i]] for idx in indices])
                weights = weights/np.sum(weights)
                confused_label = np.random.choice(indices, 1, replace=False, p=weights)[0]
                targets.append(confused_label)
                # choose an image:
                indices = np.where(self.image_labels == labels[i])[0]
                if 'iNaturalist' in train_set:
                    image_path = train_set + str(self.image_paths[np.random.choice(indices, 1)[0]])[2:-1]
                else:
                    image_path = train_set + self.image_paths[np.random.choice(indices, 1)[0]]
                images.append(Image.open(image_path).convert('RGB'))
                new_labels.append(labels[i])

        # set the models to be in eval() mode:
        for model in self.networks.values():
            model.eval()

        if self.difficulty == 0:
            possible_probs = [0.05, 0.1, 0.15]
        elif self.difficulty == 1:
            possible_probs = [0.15, 0.2, 0.25]
        elif self.difficulty == 2:
            possible_probs = [0.25, 0.3, 0.35]
        elif self.difficulty == 3:
            possible_probs = [0.5, 0.6, 0.75]
        elif self.difficulty == 4:
            possible_probs = [0.2, 0.2, 0.2]
        elif self.difficulty == 5:
            possible_probs = [0.1, 0.2, 0.3, 0.4, 0.5]

        fool = DisguisedFoolingSampleGeneration(self, self.target_confidence)
        generated_images, labels = fool.generate_batch(images, new_labels, targets, possible_probs, no_augmentation=False)

        # set the models to be in train() mode:
        for model in self.networks.values():
            model.train()
        return generated_images, labels

    def load_model(self):

        model_dir = os.path.join(self.training_opt['log_dir'], 'final_model_checkpoint'+self.name+'.pth')
        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir, map_location=self.device)
        model_state = checkpoint['state_dict_best']

        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)
        
    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, model_name):
        
        model_states = {'epoch': epoch, 'best_epoch': best_epoch, 'state_dict_best': best_model_weights,
                        'best_acc': best_acc}
        model_dir = os.path.join(self.training_opt['log_dir'], 'final_model_checkpoint'+model_name+'.pth')
        torch.save(model_states, model_dir)

