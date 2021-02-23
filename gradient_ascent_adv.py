"""
# Copyright (c) Microsoft Corporation.
# Copyright (c) Jedrzej Kozerawski
# Licensed under the MIT license.

Partially based on https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks
@author: Utku Ozbulak - github.com/utkuozbulak
Copyright (c) 2017 Utku Ozbulak
"""

import torch
from torch.optim import SGD
from torch.nn import functional
from torchvision import transforms
from torch.autograd import Variable
import numpy as np


class DisguisedFoolingSampleGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent, breaks as soon as
        the target prediction confidence is captured
    """
    def __init__(self, model, minimum_confidence):
        self.model = model
        self.minimum_confidence = minimum_confidence

    def preprocess_image(self, image, no_augmentation=False):
        if not no_augmentation:
            t = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
             t = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return Variable(t(image).unsqueeze(0), requires_grad=True)

    def generate_batch(self, images, true_labels, targets=None, possible_probs=[0.1, 0.15, 0.2], no_augmentation=False):
        n_max = 15

        batch_size = len(images)
        self.processed_images = [self.preprocess_image(images[idx], no_augmentation=no_augmentation) for idx in range(batch_size)]
        results = []
        labels = []
        
        j = 0
        confidences = np.asarray([np.random.choice(possible_probs, 1, replace=False)[0] for i in range(batch_size)])

        while batch_size > 0:

            # if it is the last pass then just return current progress:
            if j == n_max:
                for idx in range(batch_size):
                    results.append(self.images[idx])
                    labels.append(true_labels[idx])
                return results, labels

            # init optimizer with current images:
            optimizer = SGD(self.processed_images, lr=0.7)
            self.images = torch.stack(self.processed_images).squeeze(1).cuda()
            # forward pass:
            self.model.batch_forward(self.images, phase='test')
            output = self.model.logits


            # get confidence from softmax:
            target_confidences = np.asarray([functional.softmax(output)[idx][targets[idx]].data.cpu().numpy().item(0) for idx in range(batch_size)])
            # find images that already fulfill the required confidence threshold:
            good_indices = np.where(target_confidences-confidences > 0.0)[0]
            # if there are such images, then:
            if len(good_indices) > 0:
                # find ones that still require altering:
                keep_indices = [idx for idx in range(batch_size) if idx not in good_indices]
                batch_size = len(keep_indices)
  
                for idx in good_indices:
                    # add them to the final results:
                    results.append(self.images[idx])
                    labels.append(true_labels[idx])
                # remove them from next iterations:
                self.processed_images = [self.processed_images[idx] for idx in keep_indices]
                true_labels = [true_labels[idx] for idx in keep_indices]
                targets = [targets[idx] for idx in keep_indices]
                confidences = np.asarray([confidences[idx] for idx in keep_indices])

            # if no more images, return:
            if batch_size == 0:
                return results, labels
            # Target specific class
            class_loss = -output[0, targets[0]]
            for i in range(1, batch_size):
                class_loss += -output[i, targets[i]]
            # Zero grads
            self.model.networks['feat_model'].zero_grad()
            self.model.networks['classifier'].zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            j += 1
        return results, labels

