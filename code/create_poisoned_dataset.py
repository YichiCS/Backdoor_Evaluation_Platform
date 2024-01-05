import os
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets

import config
from utils import default_args, supervisor, tools

"""
Create poisoned datasets, including 
CIFAR10
BadNet, Blend, None
"""


parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, required=False, 
                    default=default_args.parser_default['dataset'], 
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False, 
                    choices=default_args.parser_choices['poison_type'], 
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float, required=False, 
                    choices=default_args.parser_choices['poison_rate'], 
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False, 
                    choices=default_args.parser_choices['cover_rate'], 
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float, required=False, 
                    default=None)
parser.add_argument('-trigger', type=str, required=False)


args = parser.parse_args()

# tools.setup_seed(0)

# settings
if args.poison_rate is None:
    args.poison_rate = config.args_default[args.dataset][args.poison_type]['poison_rate']
if args.cover_rate is None:
    args.cover_rate = config.args_default[args.dataset][args.poison_type]['cover_rate']
if args.alpha is None:
    args.alpha = config.args_default[args.dataset][args.poison_type]['alpha']
if args.trigger is None:
    args.trigger = config.args_default[args.dataset][args.poison_type]['trigger']

dataset_dir = supervisor.get_dataset_dir(args)
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

# load dataset

if args.poison_type in ['badnet', 'blend', 'cl', 'trojan', 'tact', 'adp_blend', 'adp_patch', 'wanet', 'sig', 'none']:

    if args.dataset == 'cifar10':

        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        set = datasets.CIFAR10(os.path.join(config.dataset_dir, args.dataset, 'cifar10'), train=True, download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not for now')
    
elif args.poison_type in ['dynamic']:

    if args.dataset == 'cifar10':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        set = datasets.CIFAR10(os.path.join(config.dataset_dir, args.dataset, 'cifar10'), train=True, download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
        channel_init = 32
        steps = 3
        input_channel = 3

        ckpt_path = './data/dynamic/all2one_cifar10_ckpt.pth.tar'

        normalizer = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261], [1 / 0.247, 1 / 0.243, 1 / 0.261])
        ])

    else:
        raise NotImplementedError(f'Dataset {args.dataset} not for now')
    
elif args.poison_type in ['ISSBA']:

    if args.dataset == 'cifar10':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        set = datasets.CIFAR10(os.path.join(config.dataset_dir, args.dataset, 'cifar10'), train=True, download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
        input_channel = 3

        ckpt_path = './data/ISSBA/ISSBA_cifar10.pth'
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not for now')
    

else:
    raise NotImplementedError(f'Poison_type {args.poison_type} not for now')



# choose trigger
alpha = args.alpha
poison_rate = args.poison_rate

trigger = None 
trigger_mask = None
trigger_name = args.trigger 
trigger_transform = transforms.Compose([
    transforms.ToTensor()
])
trigger_mask_transform = transforms.Compose([
    transforms.ToTensor()
])


if args.poison_type in ['badnet', 'blend', 'cl', 'trojan', 'tact', 'adp_blend', 'adp_patch', 'wanet', 'sig', 'none']:

    if trigger_name != 'none': 
        trigger_path = os.path.join(config.trigger_dir, trigger_name)
        trigger = Image.open(trigger_path).convert('RGB')
        trigger = trigger_transform(trigger)

        trigger_mask_path = os.path.join(config.trigger_dir, f'mask_{trigger_name}')

        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = trigger_mask_transform(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0), trigger[2] > 0).float()

    poison_generator = None

    if args.poison_type == 'badnet':

        from attack import badnet

        poison_generator = badnet.poison_generator(img_size=img_size, dataset=set,
                                                   poison_rate=poison_rate, trigger=trigger, trigger_mask=trigger_mask,
                                                   path=dataset_dir, target_class=config.target_class[args.dataset])
        
    elif args.poison_type == 'blend':

        from attack import blend

        poison_generator = blend.poison_generator(img_size=img_size, dataset=set,
                                                  poison_rate=args.poison_rate, trigger=trigger,
                                                  path=dataset_dir, target_class=config.target_class[args.dataset],
                                                  alpha=alpha)
    
    elif args.poison_type == 'cl':

        if args.dataset =='cifar10':
            adv_imgs_path = "./data/cl/fully_poisoned_training_datasets/two_600.npy"
            if not os.path.exists("./data/cl/fully_poisoned_training_datasets/two_600.npy"):
                raise NotImplementedError("Run 'data/cifar10/clean_label/setup.sh' first to launch clean label attack!")
            adv_imgs_src = np.load(adv_imgs_path).astype(
                np.uint8)
            adv_imgs = []
            for i in range(adv_imgs_src.shape[0]):
                adv_imgs.append(data_transform(adv_imgs_src[i]).unsqueeze(0))
            adv_imgs = torch.cat(adv_imgs, dim=0)
            assert adv_imgs.shape[0] == len(set)
        else:
            raise NotImplementedError('Clean Label Attack is not implemented for %s' % args.dataset)
        
        from attack import cl
        poison_generator = cl.poison_generator(img_size=img_size, dataset=set, adv_imgs=adv_imgs, poison_rate=poison_rate, trigger=trigger, trigger_mask=trigger_mask, path=dataset_dir, target_class=config.target_class[args.dataset])



    elif args.poison_type == 'trojan':

        from attack import trojan
        poison_generator = trojan.poison_generator(img_size=img_size, dataset=set,
                                                 poison_rate=args.poison_rate, trigger=trigger, trigger_mask=trigger_mask,
                                                 path=dataset_dir, target_class=config.target_class[args.dataset])
    
    elif args.poison_type == 'tact':

        source_class = config.args_default[args.dataset]['tact']['source_class']
        cover_classes = config.args_default[args.dataset]['tact']['cover_classes']

        from attack import tact

        poison_generator = tact.poison_generator(img_size=img_size, dataset=set,
                                                 poison_rate=args.poison_rate, cover_rate=args.cover_rate,
                                                 trigger=trigger, trigger_mask=trigger_mask,
                                                 path=dataset_dir, target_class=config.target_class[args.dataset],
                                                 source_class=source_class,
                                                 cover_classes=cover_classes)
        
    elif args.poison_type == 'adp_blend':

        from attack import adp_blend
        poison_generator = adp_blend.poison_generator(img_size=img_size, dataset=set,
                                                          poison_rate=args.poison_rate,
                                                          path=dataset_dir, trigger=trigger,
                                                          pieces=16, mask_rate=0.5,
                                                          target_class=config.target_class[args.dataset], alpha=alpha,
                                                          cover_rate=args.cover_rate)
        
    elif args.poison_type == 'adp_patch':

        trigger_names = config.args_default[args.dataset]['adp_patch']['trigger_names']

        alphas = config.args_default[args.dataset]['adp_patch']['alphas']

        from attack import adp_patch


        poison_generator = adp_patch.poison_generator(img_size=img_size, dataset=set, poison_rate=poison_rate, path=dataset_dir, trigger_names=trigger_names, alphas=alphas,target_class=config.target_class[args.dataset], cover_rate=args.cover_rate)

    elif args.poison_type == 'wanet':
        
        s = 0.5
        k = 4
        grid_rescale = 1
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            torch.nn.functional.interpolate(ins, size=img_size, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=img_size)
        x, y = torch.meshgrid(array1d, array1d, indexing='ij')
        identity_grid = torch.stack((y, x), 2)[None, ...]
        
        path = os.path.join(dataset_dir, 'identity_grid')
        torch.save(identity_grid, path)
        path = os.path.join(dataset_dir, 'noise_grid')
        torch.save(noise_grid, path)

        from attack import wanet
        poison_generator = wanet.poison_generator(img_size=img_size, dataset=set, poison_rate=args.poison_rate, cover_rate=args.cover_rate, path=dataset_dir, identity_grid=identity_grid, noise_grid=noise_grid, s=s, k=k, grid_rescale=grid_rescale,  target_class=config.target_class[args.dataset])

    elif args.poison_type == 'sig':

        delta = 30/255
        f = 6

        from attack import sig
        poison_generator = sig.poison_generator(img_size=img_size, dataset=set, poison_rate=args.poison_rate, path=dataset_dir, target_class=config.target_class[args.dataset], delta=delta, f=f)


    elif args.poison_type == 'none':

        from attack import none
        
        poison_generator = none.poison_generator(img_size=img_size, dataset=set, path=dataset_dir)


    else:
        raise NotImplementedError()

    if args.poison_type in ['badnet', 'blend', 'trojan', 'cl', 'sig', 'none']:

        img_set, poison_indices, label_set = poison_generator.generate_poisoned_set()

    elif args.poison_type in ['tact', 'adp_blend', 'adp_patch', 'wanet']:

        img_set, poison_indices, cover_indices, label_set = poison_generator.generate_poisoned_set()

        torch.save(cover_indices, os.path.join(dataset_dir, 'cover_indices'))

    else:
        raise NotImplementedError()
    

    # save the dataset as a single file 
    torch.save(img_set, os.path.join(dataset_dir, 'imgs'))
    torch.save(label_set, os.path.join(dataset_dir, 'labels'))
    torch.save(poison_indices, os.path.join(dataset_dir, 'poison_indices'))

elif args.poison_type in ['ISSBA']:

    secret_size = 20
    secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist())
    secret_path = os.path.join(dataset_dir, 'secret')
    torch.save(secret, secret_path)
    print('[Generate Poisoned Set] Save %s' % secret_path)
    
    from attack import ISSBA
    poison_generator = ISSBA.poison_generator(ckpt_path=ckpt_path, secret=secret, dataset=set, enc_height=img_size, enc_width=img_size, enc_in_channel=input_channel,
                                                poison_rate=args.poison_rate, path=dataset_dir, target_class=config.target_class[args.dataset])

    # Generate Poison Data
    img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()

    
    img_path = os.path.join(dataset_dir, 'imgs')
    label_path = os.path.join(dataset_dir, 'labels')
    poison_indices_path = os.path.join(dataset_dir, 'poison_indices')


    torch.save(img_set, img_path)
    torch.save(label_set, label_path)
    torch.save(poison_indices, poison_indices_path)

elif args.poison_type in ['dynamic']:
    if not os.path.exists(ckpt_path):
        raise NotImplementedError(f'Need {ckpt_path}')
    
    from attack import dynamic
    poison_generator = dynamic.poison_generator(ckpt_path=ckpt_path, channel_init=channel_init, steps=steps, input_channel=input_channel, normalizer=normalizer, denormalizer=denormalizer, dataset=set, poison_rate=args.poison_rate, path= dataset_dir, target_class=config.target_class[args.dataset])

    img_set, poison_indices, label_set = poison_generator.generate_poisoned_set()

    
    img_path = os.path.join(dataset_dir, 'imgs')
    label_path = os.path.join(dataset_dir, 'labels')
    poison_indices_path = os.path.join(dataset_dir, 'poison_indices')


    torch.save(img_set, img_path)
    torch.save(label_set, label_path)
    torch.save(poison_indices, poison_indices_path)



else:
    raise NotImplementedError()

    
print(f'Dataset @ {dataset_dir}')

file_name = 'info.log'
file_path = os.path.join(dataset_dir, file_name)
with open(file_path, 'w') as file:
    pass
with open(file_path, 'a') as file:
    file.write(f'[Dataset]: {args.dataset}\n')
    file.write(f'[Trigger]: {args.trigger}\n')
    file.write(f'[Poison Type]: {args.poison_type}\n')
    file.write(f'[Poison Rate]: {args.poison_rate}\n')
    file.write(f'[Cover Rate]: {args.cover_rate}\n')
    file.write(f'[Alpha]: {args.alpha}\n')
    file.write(f'[Poison Indices]: {poison_indices}\n')




