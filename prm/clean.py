
import os
import argparse

import torch
import torch.nn as nn

from torchvision import transforms

import config
from utils import default_args, tools, supervisor


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=default_args.parser_choices['poison_type'],
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-arch', type=str, required=False, default=None)
# parser.add_argument('-model_path', required=False, default=None)
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-cleanser', type=str, required=True,
                    choices=default_args.parser_choices['cleanser'])
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()

if args.poison_rate is None:
    args.poison_rate = config.args_default[args.dataset][args.poison_type]['poison_rate']
if args.cover_rate is None:
    args.cover_rate = config.args_default[args.dataset][args.poison_type]['cover_rate']
if args.alpha is None:
    args.alpha = config.args_default[args.dataset][args.poison_type]['alpha']
if args.trigger is None:
    args.trigger = config.args_default[args.dataset][args.poison_type]['trigger']
if args.arch is None:
    args.arch = config.args_default[args.dataset][args.poison_type]['arch']

if not args.test_alpha is None:
    print(f'[Attention] Test Alpha is {args.test_alpha}')

dir = supervisor.get_dataset_dir(args)
save_path = supervisor.get_cleansed_set_indices_dir(args)
cleansed = os.path.exists(save_path)
print(f'Clean @ {dir}')
arch = supervisor.get_arch(args)

if args.dataset == 'cifar10':
    num_classes = 10
else:
    raise NotImplementedError()


data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

poison_set_dir = supervisor.get_dataset_dir(args)
poisoned_set_img_path = os.path.join(poison_set_dir, 'imgs')
poisoned_set_labels_path = os.path.join(poison_set_dir, 'labels')
poisoned_set = tools.IMG_Dataset(data_path=poisoned_set_img_path, label_path=poisoned_set_labels_path, transforms=data_transform)

clean_set_dir = os.path.join(config.clean_dir, args.dataset, 'test_split')
clean_set_img_dir = os.path.join(clean_set_dir, 'imgs')
clean_set_label_path = os.path.join(clean_set_dir, 'labels')
clean_set = tools.IMG_Dataset(data_path=clean_set_img_dir, label_path=clean_set_label_path, transforms=data_transform)


if args.poison_type != 'none':
    poison_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))
else:
    poison_indices = []

model_list = []
    
args.no_aug = False
path, dir, name = supervisor.get_model_path(args)


def insepct_suspicious_indices(suspicious_indices, poison_indices, poisoned_set):
    if args.poison_type != 'none':
        true_positive  = 0
        num_positive   = len(poison_indices)
        false_positive = 0
        num_negative   = len(poisoned_set) - num_positive

        suspicious_indices.sort()
        poison_indices.sort()

        pt = 0
        for pid in suspicious_indices:
            while poison_indices[pt] < pid and pt + 1 < num_positive: pt += 1
            if poison_indices[pt] == pid:
                true_positive += 1
            else:
                false_positive += 1

        tpr = true_positive / num_positive
        fpr = false_positive / num_negative
        # if not cleansed: 
        print(f'Poisoned Model @ {path}')
        print('Elimination Rate = %d/%d = %f' % (true_positive, num_positive, tpr))
        print('Sacrifice Rate = %d/%d = %f \n' % (false_positive, num_negative, fpr))

        return tpr, fpr
    
    else:
        print(f'Clean Model @ {path}')
        false_positive = len(suspicious_indices)
        num_negative = len(poisoned_set)
        fpr = false_positive / num_negative
        print('Sacrifice Rate = %d/%d = %f \n' % (false_positive, num_negative, fpr))
        return 0, fpr
    
if args.cleanser in ['ac', 'ss', 'strip', 'sentinet', 'scan', 'ct', 'frequency']:


    model = arch(num_classes=num_classes)
    if os.path.exists(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt)
    else:
        print(f'Model {path} not exists!')

    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    suspicious_indices = []

    if args.cleanser == 'ac':
        from defense import ac
        suspicious_indices = ac.cleanser(poisoned_set, model, num_classes, args)

    elif args.cleanser == 'ss':
        from defense import ss
        suspicious_indices = ss.cleanser(poisoned_set, model, num_classes, args)

    elif args.cleanser == 'strip':
        from defense import strip
        suspicious_indices = strip.cleanser(poisoned_set, clean_set, model, args)

    elif args.cleanser == "scan":
        from defense import scan
        suspicious_indices = scan.cleanser(poisoned_set, clean_set, model, num_classes)

    elif args.cleanser == 'ct': 
        
        from defense import ct as ct
        suspicious_indices = ct.cleanser(poisoned_set, clean_set, model, num_classes, args)
            
    else:

        raise NotImplementedError()

    
    remain_indices = []

    for i in range(len(poisoned_set)):
        if i not in suspicious_indices:
            remain_indices.append(i)

    remain_indices.sort()
    torch.save(remain_indices, save_path)
    print(f'Remain indices @ {save_path}')

    tpr, fpr = insepct_suspicious_indices(suspicious_indices, poison_indices, poisoned_set)


file_name = 'clean.log'
file_path = os.path.join(dir, file_name)
with open(file_path, 'a') as file:
    file.write(f'[Cleanser]: {args.cleanser}\n')
    file.write(f'[Trigger]: {args.trigger}\n')
    file.write(f'[Poison Type]: {args.poison_type}\n')
    file.write(f'[Poison Rate]: {args.poison_rate}\n')
    file.write(f'[Cover Rate]: {args.cover_rate}\n')
    file.write(f'[Alpha]: {args.alpha}\n')
    file.write(f'[ER]: {tpr}\n')
    file.write(f'[SR]: {fpr}\n')
    file.write(f'\n')
    


