import os

import torch
import torchvision.transforms as transforms

from PIL import Image

import config

def get_cleansed_set_indices_dir(args):
        
    poison_set_dir = get_dataset_dir(args)
    dir = os.path.join(poison_set_dir, f'cleaned_indices_{args.cleanser}')

    return dir


def get_model_name(args, cleanse=False, defense=False):

    model_arch = args.arch
    poison_type = args.poison_type
    poison_rate = args.poison_rate
    cover_rate = args.cover_rate
    alpha = args.alpha
    trigger = args.trigger

    if cleanse is True:
        cleanser = args.cleanser
        if poison_type in ['badnet', 'trojan', 'cl']:
            model_name = f'{model_arch}_{cleanser}_{poison_rate:.3f}_{trigger}.pt'
            
        elif poison_type in ['blend']:
            model_name = f'{model_arch}_{cleanser}_{poison_rate:.3f}_alpha={alpha:.3f}_{trigger}.pt'

        elif poison_type in ['adp_blend']:
            model_name = f'{model_arch}_{cleanser}_{poison_rate:.3f}_cr={cover_rate:.3f}_alpha={alpha:.3f}_{trigger}.pt'

        elif poison_type in ['adp_patch', 'wanet']:
            model_name = f'{model_arch}_{cleanser}_{poison_rate:.3f}_cr={cover_rate:.3f}.pt'
        
        elif args.poison_type in ['tact']:

            model_name  = f'{model_arch}_{cleanser}_{poison_rate:.3f}_cr={cover_rate:.3f}_{trigger}.pt'

        elif poison_type in ['ISSBA', 'dynamic', 'sig']:
            model_name = f'{model_arch}_{cleanser}_{poison_rate:.3f}.pt'

        elif poison_type in ['none']:
            model_name = f'{model_arch}_{cleanser}.pt'

        else :
            raise NotImplementedError()

    else:
        if poison_type in ['badnet', 'trojan', 'cl']:
            model_name = f'{model_arch}_{poison_rate:.3f}_{trigger}.pt'
            
        elif poison_type in ['blend']:
            model_name = f'{model_arch}_{poison_rate:.3f}_alpha={alpha:.3f}_{trigger}.pt'

        elif poison_type in ['adp_blend']:
            model_name = f'{model_arch}_{poison_rate:.3f}_cr={cover_rate:.3f}_alpha={alpha:.3f}_{trigger}.pt'

        elif poison_type in ['adp_patch', 'wanet']:
            model_name = f'{model_arch}_{poison_rate:.3f}_cr={cover_rate:.3f}.pt'
        
        elif args.poison_type in ['tact']:

            model_name  = f'{model_arch}_{poison_rate:.3f}_cr={cover_rate:.3f}_{trigger}.pt'

        elif poison_type in ['ISSBA', 'dynamic', 'sig']:
            model_name = f'{model_arch}_{poison_rate:.3f}.pt'

        elif poison_type in ['none']:
            model_name = f'{model_arch}.pt'

        else :
            raise NotImplementedError()

    return model_name 

def get_model_dir(args):

    model_u_dir = os.path.join(config.model_dir, args.dataset)
    if not os.path.exists(model_u_dir):
        os.mkdir(model_u_dir)
    model_dir = os.path.join(model_u_dir, args.poison_type)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    return model_dir


def get_model_path(args, cleanse=False, defense=False):

    model_dir = get_model_dir(args)
    model_name = get_model_name(args, cleanse=cleanse, defense=defense)

    model_path = os.path.join(model_dir, model_name)

    return model_path, model_dir, model_name

def get_transforms(args):
    if args.dataset == 'cifar10':
        if args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
            ])
            data_transform = transforms.Compose([
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            data_transform = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                     [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])

    else :
        raise NotImplementedError()

    return data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer

def get_arch(args):
    if args.arch not in config.arch[args.dataset]:
            raise NotImplementedError()
    else:
        arch =  args.arch
    return config.arch_map[arch]
        

def get_dataset_dir(args, split='train'):

    if split == 'train':
        if args.poison_type in ['badnet', 'trojan', 'cl']:

            dir = f'{args.poison_type}_{args.poison_rate:.3f}_{args.trigger}'

        elif args.poison_type in ['blend']:

            dir = f'{args.poison_type}_{args.poison_rate:.3f}_alpha={args.alpha}_{args.trigger}'

        elif args.poison_type in ['adp_blend']:

            dir = f'{args.poison_type}_{args.poison_rate:.3f}_cr={args.cover_rate:.3f}_alpha={args.alpha}_{args.trigger}'

        elif args.poison_type in ['adp_patch', 'wanet']:

            dir = f'{args.poison_type}_{args.poison_rate:.3f}_cr={args.cover_rate:.3f}'

        elif args.poison_type in ['tact']:

            dir = f'{args.poison_type}_{args.poison_rate:.3f}_cr={args.cover_rate:.3f}_{args.trigger}'

        elif args.poison_type in ['ISSBA', 'dynamic', 'sig']:
            
            dir = f'{args.poison_type}_{args.poison_rate:.3f}'
        
        elif args.poison_type in ['none']:

            dir = f'{args.poison_type}'

        else:
            raise NotImplementedError()
            
        if not os.path.exists(os.path.join(config.dataset_dir, args.dataset)):
            os.mkdir(os.path.join(config.dataset_dir, args.dataset))

        dataset_dir = os.path.join(config.dataset_dir, args.dataset, dir)
            
    elif split == 'test':

        dir = f'test_split'
        dataset_dir = os.path.join(config.clean_dir, args.dataset, dir)

    else:
        raise NotImplementedError('Wrong Split')

    return dataset_dir

def get_trigger(trigger_name, trigger_transform, trigger_mask_transform):

    if trigger_name != None:
        trigger_path = os.path.join(config.trigger_dir, trigger_name)
        trigger = Image.open(trigger_path).convert("RGB")

        trigger_mask_path = os.path.join(config.trigger_dir, f'mask_{trigger_name}')

        if os.path.exists(trigger_mask_path):
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = trigger_mask_transform(trigger_mask)[0]
        else:
            trigger_map = trigger_mask_transform(trigger)
            trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0), trigger_map[2] > 0).float()

        trigger = trigger_transform(trigger)
        trigger_mask = trigger_mask

    else:
        return None, None

    return trigger, trigger_mask



def get_poison_transform(args, target_class, trigger_transform, is_normalized_input=False):

    alpha = args.alpha if args.test_alpha is None else args.test_alpha

    trigger_name=args.trigger

    if args.dataset in ['cifar10']:
        img_size = 32
    else:
        raise NotImplementedError()

    if args.dataset == 'cifar10':
        normalizer = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                    [1 / 0.247, 1 / 0.243, 1 / 0.261])
        ])
        num_classes = 10

    else:
        raise NotImplementedError()

    poison_transform = None
    trigger = None
    trigger_mask = None


    if args.poison_type in ['badnet', 'blend', 'trojan', 'cl', 'tact', 'adp_blend', 'adp_patch', 'wanet', 'sig']:

        trigger_mask_transform_list = []
        for t in trigger_transform.transforms:
            if "Normalize" not in t.__class__.__name__:
                trigger_mask_transform_list.append(t)
        trigger_mask_transform = transforms.Compose(trigger_mask_transform_list)

        if trigger_name != 'none':
            trigger_path = os.path.join(config.trigger_dir, trigger_name)
            trigger = Image.open(trigger_path).convert("RGB")
            
            
            trigger_mask_path = os.path.join(config.trigger_dir, f'mask_{trigger_name}')
            if os.path.exists(trigger_mask_path):
                    trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                    # trigger_mask = transforms.ToTensor()(trigger_mask)
                    trigger_mask = trigger_mask_transform(trigger_mask)[0]
            else:
                trigger_map = trigger_mask_transform(trigger)
                
                trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0), trigger_map[2] > 0).float()
        
        if trigger is not None:
            trigger = trigger_transform(trigger)

        trigger_mask = trigger_mask

        if args.poison_type == 'badnet':
            from attack import badnet
            poison_transform = badnet.poison_transform(img_size=img_size, trigger=trigger,
                                                        trigger_mask=trigger_mask, target_class=target_class)
            
        elif args.poison_type == 'blend':
            from attack import blend
            poison_transform = blend.poison_transform(img_size=img_size, trigger=trigger,
                                                        target_class=target_class, alpha=alpha)
        
        elif args.poison_type == 'trojan':
            from attack import trojan
            poison_transform = trojan.poison_transform(img_size=img_size, trigger=trigger,
                                                        trigger_mask=trigger_mask, target_class=target_class)
            
        elif args.poison_type == 'cl':
            from attack import cl
            poison_transform = cl.poison_transform(img_size=img_size, trigger=trigger,
                                                        trigger_mask=trigger_mask, target_class=target_class)
            
        elif args.poison_type == 'tact':

            from attack import tact
            poison_transform = tact.poison_transform(img_size=img_size, trigger=trigger,
                                                        trigger_mask=trigger_mask, target_class=target_class)
        elif args.poison_type == 'adp_blend':

            from attack import adp_blend
            poison_transform = adp_blend.poison_transform(img_size=img_size, trigger=trigger,
                                                               target_class=target_class, alpha=alpha)
            
        elif args.poison_type == 'adp_patch':

            test_trigger_names = config.args_default[args.dataset]['adp_patch']['test_trigger_names']

            test_alphas = config.args_default[args.dataset]['adp_patch']['test_alphas']

            from attack import adp_patch
            poison_transform = adp_patch.poison_transform(img_size=img_size, trigger_names=test_trigger_names, alphas=test_alphas, target_class=target_class, normalizer=normalizer, denormalizer=denormalizer)

        elif args.poison_type == 'wanet':
            s = 0.5
            k = 4
            grid_rescale = 1
            path = os.path.join(get_dataset_dir(args), 'identity_grid')
            identity_grid = torch.load(path)
            path = os.path.join(get_dataset_dir(args), 'noise_grid')
            noise_grid = torch.load(path)

            from attack import wanet
            poison_transform = wanet.poison_transform(img_size=img_size, denormalizer=denormalizer, identity_grid=identity_grid, noise_grid=noise_grid, s=s, k=k, grid_rescale=grid_rescale, normalizer=normalizer, target_class=target_class)

        elif args.poison_type == 'sig':

            delta = 30/255
            f = 6

            from attack import sig
            poison_transform = sig.poison_transform(img_size=img_size, denormalizer=denormalizer, normalizer=normalizer, target_class=target_class, delta=delta, f=f, has_normalized=is_normalized_input)
            

        elif args.poison_type == 'none':
            from attack import none
            poison_transform = none.poison_transform()

        else:
            raise NotImplementedError()
        
        return poison_transform

    elif args.poison_type in ['ISSBA']:

        if args.dataset == 'cifar10':
            ckpt_path = './data/ISSBA/ISSBA_cifar10.pth'
            input_channel = 3
            img_size = 32

        else :
            raise NotImplementedError()
        

        if not os.path.exists(ckpt_path):
            raise NotImplementedError(
                'Need ISSBA_cifar10.pth')

        secret_path = os.path.join(get_dataset_dir(args), 'secret')
        secret = torch.load(secret_path)

        from attack import ISSBA
        poison_transform = ISSBA.poison_transform(ckpt_path=ckpt_path, secret=secret, normalizer=normalizer,
                                                  denormalizer=denormalizer,
                                                  enc_in_channel=input_channel, enc_height=img_size, enc_width=img_size,
                                                  target_class=target_class)
        return poison_transform
    
    elif args.poison_type in ['dynamic']:

        if args.dataset == 'cifar10':
            channel_init = 32
            steps = 3
            input_channel = 3
            ckpt_path = './data/dynamic/all2one_cifar10_ckpt.pth.tar'

            require_normalization = True

        else:
            raise NotImplementedError()
        
        if not os.path.exists(ckpt_path):
            raise NotImplementedError(
                f'Need {ckpt_path}')
        
        from attack import dynamic
        poison_transform = dynamic.poison_transform(ckpt_path=ckpt_path, channel_init=channel_init, steps=steps,
                                                    input_channel=input_channel, normalizer=normalizer,
                                                    denormalizer=denormalizer, target_class=target_class,
                                                    has_normalized=is_normalized_input,
                                                    require_normalization=require_normalization)
        return poison_transform



    else:
        raise NotImplementedError()
