import os
import argparse

import torch
import torch.nn as nn
import config

from utils import default_args, supervisor, tools

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
parser.add_argument('-alpha', type=float, required=False, default=None)
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False, default=None)
parser.add_argument('-arch', type=str, required=False, default=None)
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
parser.add_argument('-cleanser', type=str, required=False,
                    choices=default_args.parser_choices['cleanser'])

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


data_transform_aug, data_transform, trigger_transform, _, _ = supervisor.get_transforms(args)


if args.dataset == 'cifar10':
    num_classes = 10
    batch_size = 128

else:
    raise NotImplementedError('dataset not for now')

cleanse = False if args.cleanser is None else True

test_set_dir = supervisor.get_dataset_dir(args, split='test')
model_path, _, _ = supervisor.get_model_path(args, cleanse=cleanse)

# print(args.cleanser, model_path)
# exit(0)

print(f'Model @ {model_path}')
print(f'Testset @ {test_set_dir}')


arch = supervisor.get_arch(args)

# model_path = '/data/yichi/BF/akl/models/cifar10/badnet/resnet_0.003_badnet_patch_32.png.pt'
model = arch(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model = nn.DataParallel(model)
model = model.cuda()

if args.dataset in ['cifar10']:
    
    test_set_img_path = os.path.join(test_set_dir, 'imgs')
    test_set_label_path = os.path.join(test_set_dir, 'labels')

    
    # test_set_img_path = '/data/yichi/BF/akl/clean/cifar10/test_split/imgs'
    # test_set_label_path = '/data/yichi/BF/akl/clean/cifar10/test_split/labels'


    test_set = tools.IMG_Dataset(data_path=test_set_img_path, label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    poison_transform = supervisor.get_poison_transform(args, target_class=config.target_class[args.dataset], trigger_transform=trigger_transform)

else:
    print('#TODO2')


if args.poison_type not in ['tact']:

    source_classes = None

else:
    source_classes = [config.args_default[args.dataset]['tact']['source_class']]

tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes)
















