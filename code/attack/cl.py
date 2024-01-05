import os
import torch
import random

from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, adv_imgs, poison_rate, trigger, trigger_mask, path, target_class):

        self.img_size = img_size
        self.dataset = dataset
        self.adv_imgs = adv_imgs
        self.poison_rate = poison_rate

        self.trigger = trigger 
        self.trigger_mask = trigger_mask
        self.path = path 
        self.target_class = target_class

        self.dx, self.dy = trigger_mask.shape

        self.num_img = len(dataset)

    def generate_poisoned_set(self):

        all_target_indices = []
        all_other_indices = []
        for i in range(self.num_img):
            _, gt = self.dataset[i]
            if gt == self.target_class:
                all_target_indices.append(i)
            else:
                all_other_indices.append(i)

        random.shuffle(all_target_indices)
        random.shuffle(all_other_indices)

        num_poison = int(self.num_img * self.poison_rate)

        poison_indices = all_target_indices[:num_poison]
        poison_indices.sort()

        img_set = []
        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                img = self.adv_imgs[i]

                img = img + self.trigger_mask * (self.trigger - img)

                os.makedirs(os.path.join(self.path, 'poisoned_img'), exist_ok=True)
                save_image(img, os.path.join(self.path, 'poisoned_img', f'label_{i}.png'))

                pt += 1

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, poison_indices, label_set


class poison_transform():

    def __init__(self, img_size, trigger, trigger_mask, target_class=0):
        self.img_size = img_size
        self.target_class = target_class 
        self.trigger = trigger
        self.trigger_mask = trigger_mask
        self.dx, self.dy = trigger_mask.shape

    def transform(self, data, labels):

        data, labels = data.clone(), labels.clone()

        labels[:] = self.target_class

        data = data + self.trigger_mask.to(data.device) * (self.trigger.to(data.device) - data)

        return data, labels
