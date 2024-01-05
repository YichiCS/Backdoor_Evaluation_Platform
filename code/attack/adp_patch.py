import os
import torch
import random

from torchvision.utils import save_image

import config
from torchvision import transforms
from PIL import Image

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger_names, alphas, target_class=0, cover_rate=0.01):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path 
        self.target_class = target_class
        self.cover_rate = cover_rate

        self.num_img = len(dataset)

        trigger_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.triggers = []
        self.trigger_masks = []
        self.alphas = []

        for i in range(len(trigger_names)):
            trigger_path = os.path.join(config.trigger_dir, trigger_names[i])
            trigger_mask_path = os.path.join(config.trigger_dir, f'mask_{trigger_names[i]}')

            trigger = Image.open(trigger_path).convert('RGB')
            trigger = trigger_transform(trigger)

            if os.path.exists(trigger_mask_path):  
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]
            else:  
                trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),trigger[2] > 0).float()

            self.triggers.append(trigger)
            self.trigger_masks.append(trigger_mask)
            self.alphas.append(alphas[i])

    def generate_poisoned_set(self):

        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover]
        cover_indices.sort()

        img_set = []
        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        poison_id = []
        cover_id = []

        k = len(self.triggers)

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                for j in range(k):
                    if ct < (j+1) * (num_cover/k):
                        img  = img + self.alphas[j] * self.trigger_masks[j] * (self.triggers[j] - img)

                        break
                
                ct += 1

            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class

                for j in range(k):
                    if pt < (j + 1) * (num_poison / k):
                        img = img + self.alphas[j] * self.trigger_masks[j] * (self.triggers[j] - img)
                        
                        break
                pt += 1

                os.makedirs(os.path.join(self.path, 'poisoned_img'), exist_ok=True)
                save_image(img, os.path.join(self.path, 'poisoned_img', f'label_{i}.png'))

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt += 1

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id

        return img_set, poison_indices, cover_indices, label_set
    
class poison_transform():

    def __init__(self, img_size, trigger_names, alphas, target_class=0, denormalizer=None, normalizer=None):

        self.img_size = img_size
        self.target_class = target_class
        self.denormalizer = denormalizer
        self.normalizer = normalizer

        # triggers
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.triggers = []
        self.trigger_masks = []
        self.alphas = []

        for i in range(len(trigger_names)):
            trigger_path = os.path.join(config.trigger_dir, trigger_names[i])
            trigger_mask_path = os.path.join(config.trigger_dir, f'mask_{trigger_names[i]}')
            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)

            if os.path.exists(trigger_mask_path):  

                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]  
            else:  
                trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0), trigger[2] > 0).float()

            self.triggers.append(trigger.cuda())
            self.trigger_masks.append(trigger_mask.cuda())
            self.alphas.append(alphas[i])

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()

        data = self.denormalizer(data)
        for j in range(len(self.triggers)):
            data = data + self.alphas[j] * self.trigger_masks[j].to(data.device) * (self.triggers[j].to(data.device) - data)
        data = self.normalizer(data)

        labels[:] = self.target_class


        return data, labels