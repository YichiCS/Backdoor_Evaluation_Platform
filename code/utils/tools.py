import os
import random


from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn

from torchvision import transforms, datasets
from torch.utils.data import DataLoader


import numpy as np

from torch.utils.data import Dataset
import utils.supervisor as supervisor

class IMG_Dataset(Dataset):
    def __init__ (self, data_path, label_path, num_class=10, transforms=None, shift=False, random_labels=False):

        self.dir = data_path 
        self.img_set = torch.load(data_path)
        self.gt = torch.load(label_path)
        self.transforms = transforms
        self.num_class = num_class
        self.shift = shift
        self.random_labels = random_labels

    def __len__(self):

        return len(self.gt)
    
    def __getitem__(self, index):

        index = int(index)

        img = self.img_set[index]

        if self.transforms is not None:
            img = self.transforms(img)

        
        if self.random_labels:
            label = torch.randint(self.num_classes,(1,))[0]
        else:
            label = self.gt[index]
            if self.shift:
                label = (label+1) % self.num_classes

    
        return img, label
   

def test(model, test_loader, poison_test = False, poison_transform=None, source_classes=None, num_classes=10):

    model.eval()
    clean_correct = 0
    poison_correct = 0
    tot = 0
    num_non_target_class = 0
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0
    poison_acc = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):

            data, target = data.cuda(), target.cuda()
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += clean_pred.eq(target).sum().item()

            tot += len(target)
            this_batch_size = len(target)
            tot_loss += criterion(clean_output, target) * this_batch_size



            if poison_test:
                clean_target = target
                data, target = poison_transform.transform(data, target)

                poison_output = model(data)
                poison_pred = poison_output.argmax(dim=1, keepdim=True)

                target_class = target[0].item()
                for bid in range(this_batch_size):
                    if clean_target[bid]!=target_class:
                        if source_classes is None:
                            num_non_target_class+=1
                            if poison_pred[bid] == target_class:
                                poison_correct+=1
                        else: # for source-specific attack
                            if clean_target[bid] in source_classes:
                                num_non_target_class+=1
                                if poison_pred[bid] == target_class:
                                    poison_correct+=1

                poison_acc += poison_pred.eq((clean_target.view_as(poison_pred))).sum().item()
    
    print('Clean ACC: {}/{} = {:.6f}, Loss: {}'.format(
            clean_correct, tot,
            clean_correct/tot, tot_loss/tot
    ))

    if poison_test:
        print('ASR: %d/%d = %.6f' % (poison_correct, num_non_target_class, poison_correct / num_non_target_class))
        return clean_correct/tot, poison_correct / num_non_target_class
    
    return clean_correct/tot, None

def worker_init(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

