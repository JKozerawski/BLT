"""
# Copyright (c) Microsoft Corporation.
# Copyright (c) Jedrzej Kozerawski
# Licensed under the MIT license.

Partially based on https://github.com/zhmiao/OpenLongTailRecognition-OLTR
@author: Zhongqi Miao
Copyright (c) 2019, Zhongqi Miao
# Licensed under the BSD 3-Clause License.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import importlib
import pdb

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def print_write(print_str, log_file):
    print(*print_str)
    #with open(log_file, 'a') as f:
        #print(*print_str, file=f)

def init_weights(model, weights_path, caffe=False, classifier=False, temp=False, device = 'cuda:1'):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))    
    weights = torch.load(weights_path, map_location=device)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
    else:      
        weights = weights['state_dict_best']['classifier']
        if not temp:
            weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
        else:
            return weights
    model.load_state_dict(weights)   
    return model

def correlation(accuracies, counts):
    num_images = np.sum(counts)
    balanced_counts = np.asarray([num_images/len(counts) for i in range(len(counts))])
    few_shot_indices = np.where(counts < 20)[0]
    hallucinated_counts = np.asarray([num_images/len(counts) for i in range(len(counts))])
    for idx in few_shot_indices:
        hallucinated_counts[idx] *= 1.5
    #print(balanced_counts)
    #print(hallucinated_counts)
    accuracy_correlation = np.corrcoef(accuracies, counts)
    balanced_correlation = np.corrcoef(accuracies, balanced_counts)
    hallucinated_correlation = np.corrcoef(accuracies,  hallucinated_counts)
    #print("Correlations:", accuracy_correlation, balanced_correlation, hallucinated_correlation)
    print("Correlations:", accuracy_correlation[0, 1],  balanced_correlation[0, 1], hallucinated_correlation[0, 1])
    #return

def shot_acc (correct, labels, label_groups, counts):

    tlabels = labels.detach().cpu().numpy()
    unique = np.unique(tlabels)
    class_correct_1 = np.zeros(len(unique))
    class_correct_5 = np.zeros(len(unique))
    many_shot_1 = []
    medium_shot_1 = []
    low_shot_1 = []
    many_shot_5 = []
    medium_shot_5 = []
    low_shot_5 = []
    for l in unique:
        n = len(np.where(tlabels == l)[0])
        #if n == 0:
            #n = 1
        correct_k = correct[:1, labels == l].view(-1).float().sum(0, keepdim=True).cpu().numpy()[0]
        class_correct_1[l] = correct_k / n
        correct_k = correct[:5, labels == l].view(-1).float().sum(0, keepdim=True).cpu().numpy()[0]
        class_correct_5[l] = correct_k / n
        if l in label_groups[0]:
            many_shot_1.append(class_correct_1[l])
            many_shot_5.append(class_correct_5[l])
        elif l in label_groups[2]:
            low_shot_1.append(class_correct_1[l])
            low_shot_5.append(class_correct_5[l])
        else:
            medium_shot_1.append(class_correct_1[l])
            medium_shot_5.append(class_correct_5[l])
    correlation(class_correct_1, counts)
    print("Standard deviations:", 100*np.std(many_shot_1), 100*np.std(medium_shot_1), 100*np.std(low_shot_1))

    return 100*np.mean(many_shot_1), 100*np.mean(medium_shot_1), 100*np.mean(low_shot_1), 100*np.mean(many_shot_5), 100*np.mean(medium_shot_5), 100*np.mean(low_shot_5)


def shot_acc_old(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20):
    training_labels = np.array(train_data.dataset.labels).astype(int)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def shot_acc2(preds, labels, train_data, label_groups):
    training_labels = np.array(train_data.dataset.labels).astype(int)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    medium_shot = []
    low_shot = []

    many_count = 0.
    medium_count = 0.
    few_count = 0.
    for i in range(len(train_class_count)):
        if i in label_groups[0]:
            many_shot.append((class_correct[i] / test_class_count[i]))
            many_count += 1.
        elif i in label_groups[2]:
            low_shot.append((class_correct[i] / test_class_count[i]))
            few_count += 1.
        else:
            medium_shot.append((class_correct[i] / test_class_count[i]))
            medium_count += 1.

    many_err = round((1. / np.sqrt(many_count)) * np.std(many_shot), 3)
    medium_err = round((1. / np.sqrt(medium_count)) * np.std(medium_shot), 3)
    few_err = round((1. / np.sqrt(few_count)) * np.std(low_shot), 3)

    return np.mean(many_shot), np.mean(medium_shot), np.mean(low_shot), many_err, medium_err, few_err
        
def F_measure(preds, labels, openset=False, theta=None):
    
    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.
        
        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 and preds[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')

def mic_acc_cal(preds, labels):
    acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return 100*acc_mic_top1

def class_count (data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num

# def dataset_dist (in_loader):

#     """Example, dataset_dist(data['train'][0])"""
    
#     label_list = np.array([x[1] for x in in_loader.dataset.samples])
#     total_num = len(data_list)

#     distribution = []
#     for l in np.unique(label_list):
#         distribution.append((l, len(label_list[label_list == l])/total_num))
        
#     return distribution

