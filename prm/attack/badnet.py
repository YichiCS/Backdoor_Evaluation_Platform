import os
import torch
import random

from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger, trigger_mask, target_class=0, alpha=1.0):
        
        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path
        self.trigger = trigger
        self.trigger_mask = trigger_mask
        self.target_class = target_class
        self.alpha = alpha

        self.num_img = len(dataset)

    def generate_poisoned_set(self):

        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()

        img_set = []
        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:

                gt = self.target_class
                pt+=1

                img = img + self.alpha * self.trigger_mask * (self.trigger - img)
                
                os.makedirs(os.path.join(self.path, 'poisoned_img'), exist_ok=True)
                save_image(img, os.path.join(self.path, 'poisoned_img', f'label_{i}.png'))


            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, poison_indices, label_set
    
class poison_transform():
    def __init__(self, img_size, trigger, trigger_mask, target_class=0, alpha=1.0):
        self.img_size = img_size
        self.target_class = target_class
        self.trigger = trigger
        self.trigger_mask = trigger_mask
        self.alpha = alpha

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = data + self.alpha * self.trigger_mask.to(data.device) * (self.trigger.to(data.device) - data)
        labels[:] = self.target_class

        return data, labels

