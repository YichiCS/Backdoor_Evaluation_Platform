import os
import torch
import random
from torchvision.utils import save_image
from math import sqrt

def issquare(x):
    tmp = sqrt(x)
    tmp2 = round(tmp)
    return abs(tmp - tmp2) <= 1e-8


def get_trigger_mask(img_size, total_pieces, masked_pieces):
    div_num = sqrt(total_pieces)
    step = int(img_size // div_num)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    mask = torch.ones((img_size, img_size))
    for i in candidate_idx:
        x = int(i % div_num)
        y = int(i // div_num)
        mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0

    return mask


class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger, target_class=0, alpha=0.2, cover_rate=0.01,
                 pieces=16, mask_rate=0.5):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path 
        self.target_class = target_class 
        
        self.trigger = trigger
        self.alpha = alpha
        self.cover_rate = cover_rate
        assert abs(round(sqrt(pieces)) - sqrt(pieces)) <= 1e-8
        assert img_size % round(sqrt(pieces)) == 0
        self.pieces = pieces
        self.mask_rate = mask_rate
        self.masked_pieces = round(self.mask_rate * self.pieces)

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_set(self):

        # random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() 

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover] 
        
        cover_indices.sort()

        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        img_set = []
        poison_id = []
        cover_id = []

        for i in range(self.num_img):
            img, gt = self.dataset[i]


            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
                img = img + self.alpha * mask * (self.trigger - img)
                ct += 1

            if pt < num_poison and poison_indices[pt] == i:

                poison_id.append(cnt)
                gt = self.target_class
                mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
                img = img + self.alpha * mask * (self.trigger - img)
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

    def __init__(self, img_size, trigger, target_class=0, alpha=0.2):
        self.img_size = img_size
        self.target_class = target_class
        self.trigger = trigger
        self.alpha = alpha

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = data + self.alpha * (self.trigger.to(data.device) - data)
        labels[:] = self.target_class


        return data, labels