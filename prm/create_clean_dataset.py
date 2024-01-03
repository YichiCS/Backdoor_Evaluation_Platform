import os
import random
import argparse

import torch
from PIL import Image
from torchvision import transforms, datasets

import config
from utils import default_args, supervisor, tools

from torchvision.utils import save_image

"""
Create poisoned datasets, including 
CIFAR10
BadNet, Blend, None
"""


parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, required=False, 
                    default=default_args.parser_default['dataset'], 
                    choices=default_args.parser_choices['dataset'])

args = parser.parse_args()

tools.setup_seed(0)

# settings
clean_dir = config.clean_dir
dataset_dir = os.path.join(clean_dir, args.dataset)
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

# load dataset
if args.dataset == 'cifar10':

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = datasets.CIFAR10(os.path.join(dataset_dir, 'cifar10'),
                                train=True, download=True, transform=data_transform)
    test_set = datasets.CIFAR10(os.path.join(dataset_dir, 'cifar10'),
                                train=False, download=True, transform=data_transform)
    img_size = 32
    num_classes = 10
else:
    print(f'<Undefined> Dataset = {args.dataset} ')
    exit(0)

train_dir = os.path.join(dataset_dir, 'train_split')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

test_dir = os.path.join(dataset_dir, 'test_split')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# Train Split

num_img = len(train_set)
id_set = list(range(0, num_img))
train_img_dir = os.path.join(train_dir, 'img')
if not os.path.exists(train_img_dir):
    os.mkdir(train_img_dir)

img_set = []
label_set = []

for i in range(num_img):
    img, gt = train_set[i]
    img_name = f'{i}.png'
    img_path = os.path.join(train_img_dir, img_name)
    save_image(img, img_path)

    img_set.append(img)
    label_set.append(gt)

img_set = torch.cat(img_set, dim=0)
label_set = torch.LongTensor(label_set)

torch.save(img_set, os.path.join(train_dir, 'imgs'))
torch.save(label_set, os.path.join(train_dir, 'labels'))



# Test Split

num_img = len(test_set)
id_set = list(range(0, num_img))
test_img_dir = os.path.join(test_dir, 'data')
if not os.path.exists(test_img_dir):
    os.mkdir(test_img_dir)

img_set = []
label_set = []

for i in range(num_img):
    img, gt = test_set[i]
    # print(img.shape)
    img_name = f'{i}.png'
    img_path = os.path.join(test_img_dir, img_name)
    save_image(img, img_path)

    img_set.append(img.unsqueeze(0))
    label_set.append(gt)

img_set = torch.cat(img_set, dim=0)
label_set = torch.LongTensor(label_set)

torch.save(img_set, os.path.join(test_dir, 'imgs'))
torch.save(label_set, os.path.join(test_dir, 'labels'))


print(f'Dataset @ {dataset_dir}')







