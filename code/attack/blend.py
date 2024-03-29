import torch
import random

from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, trigger, path, target_class=0, alpha=0.2):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path
        self.trigger = trigger
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

                img = (1 - self.alpha) * img + self.alpha * self.trigger
                
                import os
                os.makedirs(os.path.join(self.path, 'poisoned_img'), exist_ok=True)
                save_image(img, os.path.join(self.path, 'poisoned_img', f'label_{i}.png'))


            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, poison_indices, label_set

class poison_transform():
    def __init__(self, img_size, trigger, target_class=0, alpha=0.2):

        self.img_size = img_size
        self.trigger = trigger
        self.target_class = target_class
        self.alpha = alpha

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = (1 - self.alpha) * data + self.alpha * self.trigger.to(data.device)
        labels[:] = self.target_class

        return data, labels
