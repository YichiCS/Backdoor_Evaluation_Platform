import os
import torch
from torchvision import transforms

from utils.arch import resnet
from utils import supervisor
from utils import tools

num_workers = 4

clean_dir = './clean'
dataset_dir = './datasets'
trigger_dir = './triggers'
model_dir = './models'

test_freq = 0.05
poison_seed = 0

args_default = {
    'cifar10': {
        'none': {
            'poison_rate': 0,
            'cover_rate': 0,
            'alpha': 0,
            'trigger' : 'none',
            'arch': 'resnet',
        },
        'badnet': {
            'poison_rate': 0.003,
            'cover_rate': 0,
            'alpha': 1.0,
            'trigger' : 'badnet_patch_32.png',
            'arch': 'resnet',
        },
        'blend': {
            'poison_rate': 0.003,
            'cover_rate': 0,
            'alpha': 0.2,
            'trigger' : 'hellokitty_32.png',
            'arch': 'resnet',
        },
        'trojan': {
            'poison_rate': 0.003,
            'cover_rate': 0,
            'alpha': 1.0, 
            'trigger' : 'trojan_square_32.png',
            'arch': 'resnet',
        },
        'cl': {
            'poison_rate': 0.003,
            'cover_rate': 0,
            'alpha': 1.0, 
            'trigger' : 'badnet_patch4_dup_32.png',
            'arch': 'resnet',
        },
        'ISSBA': {
            'poison_rate': 0.02,
            'cover_rate': 0,
            'alpha': 1.0, 
            'trigger' : 'badnet_patch_32.png',
            'arch': 'resnet',
        },
        'dynamic': {
            'poison_rate': 0.003,
            'cover_rate': 0,
            'alpha': 1.0, 
            'trigger' : 'badnet_patch_32.png',
            'arch': 'resnet',
        },
        'tact': {
            'poison_rate': 0.003,
            'cover_rate': 0.003,
            'alpha': 1.0, 
            'trigger' : 'trojan_square_32.png',
            'arch': 'resnet',
            'source_class': 1,
            'cover_classes': [5, 7],
        },
        'adp_blend': {
            'poison_rate': 0.003,
            'cover_rate': 0.003,
            'alpha': 0.15, 
            'trigger' : 'hellokitty_32.png',
            'arch': 'resnet',
        },
        'adp_patch': {
            'poison_rate': 0.003,
            'cover_rate': 0.006,
            'alpha': 1.0, 
            'trigger' : 'none',
            'arch': 'resnet',
            'trigger_names':[
                'phoenix_corner_32.png',
                'firefox_corner_32.png',
                'badnet_patch4_32.png',
                'trojan_square_32.png',],
            'alphas':[
                0.5,
                0.2,
                0.5,
                0.3,],
            'test_trigger_names':[
                'phoenix_corner2_32.png',
                'badnet_patch4_32.png',],
            'test_alphas':[
                1,
                1],
        },
        'wanet': {
            'poison_rate': 0.050,
            'cover_rate': 0.100,
            'alpha': 1.0, 
            'trigger' : 'none',
            'arch': 'resnet',
        },
        'sig': {
            'poison_rate': 0.02,
            'cover_rate': 0,
            'alpha': 1.0, 
            'trigger' : 'none',
            'arch': 'resnet',
        },
    },
}

target_class = {
    'cifar10' : 0,
}

arch = {
    'cifar10' : ['resnet'],

}

arch_map = {
    'resnet' : resnet.ResNet18
}


def get_params(args):

    if args.dataset == 'cifar10':

        num_classes = 10

        data_transform_normalize = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])

        data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            # transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])

        distillation_ratio = [1/2, 1/5, 1/25, 1/50, 1/100]
        momentums = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        lambs = [20, 20, 20, 30, 30, 15]
        lrs = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01]
        batch_factors = [2, 2, 2, 2, 2, 2]

    else:
        raise NotImplementedError('<Unimplemented Dataset> %s' % args.dataset)


    params = {
        'data_transform' : data_transform_normalize,
        'data_transform_aug' : data_transform_aug,

        'distillation_ratio': distillation_ratio,
        'momentums': momentums,
        'lambs': lambs,
        'lrs': lrs,
        'lr_base': 0.1,
        'batch_factors': batch_factors,
        'weight_decay' : 1e-4,
        'num_classes' : num_classes,
        'batch_size' : 128,
        # 'pretrain_epochs' : 100,
        'pretrain_epochs' : 100,
        'median_sample_rate': 0.1,
        'base_arch' :  arch_map[arch[args.dataset][0]],
        'arch' :  arch_map[arch[args.dataset][0]],
        'kwargs' : {'num_workers': 2, 'pin_memory': True},
        'inspection_set_dir': supervisor.get_dataset_dir(args)
    }


    return params
