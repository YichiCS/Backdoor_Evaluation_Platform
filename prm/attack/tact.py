import os
import torch
import random

from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, cover_rate, trigger, trigger_mask, path, target_class=0, source_class=1, cover_classes=[5,7]):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate
        self.trigger = trigger
        self.trigger_mask = trigger_mask
        self.path = path
        self.target_class = target_class
        self.source_class= source_class
        self.cover_classes = cover_classes
        self.num_img = len(dataset)

    def generate_poisoned_set(self):

        all_source_indices = []
        all_cover_indices = []

        for i in range(self.num_img):
            _, gt = self.dataset[i]

            if gt == self.source_class:
                all_source_indices.append(i)
            elif gt in self.cover_classes:
                all_cover_indices.append(i)


        random.shuffle(all_source_indices)
        random.shuffle(all_cover_indices)

        num_poison = int(self.num_img * self.poison_rate)
        num_cover = int(self.num_img * self.cover_rate)

        poison_indices = all_source_indices[:num_poison]
        cover_indices = all_cover_indices[:num_cover]

        poison_indices.sort()
        cover_indices.sort()


        img_set = []
        label_set = []

        pt = 0
        ct = 0

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                img = (1 - self.trigger_mask) * img + self.trigger_mask * self.trigger
                pt += 1 

                os.makedirs(os.path.join(self.path, 'poisoned_img'), exist_ok=True)
                save_image(img, os.path.join(self.path, 'poisoned_img', f'label_{i}.png'))


            if ct < num_cover and cover_indices[ct] == i:
                img = (1 - self.trigger_mask) * img + self.trigger_mask * self.trigger
                ct += 1

            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)


        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, poison_indices, cover_indices, label_set
    
class poison_transform():
    
    def __init__(self, img_size, trigger, trigger_mask, target_class = 0):
        self.img_size = img_size
        self.trigger = trigger
        self.trigger_mask = trigger_mask
        self.target_class = target_class # by default : target_class = 0

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()

        labels[:] = self.target_class
        data = (1 - self.trigger_mask.to(data.device)) * data + self.trigger_mask.to(data.device) * self.trigger.to(data.device)

        return data, labels
            








