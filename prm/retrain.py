
import os
import time
import argparse

from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn

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
parser.add_argument('-model_path', required=False, default=None)

parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-cleanser', type=str, required=True,
                    choices=default_args.parser_choices['cleanser'])
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

parser.add_argument('-akl', default=False, action='store_true')

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


batch_size = 128
model_path, model_dir, model_name = supervisor.get_model_path(args, cleanse=True)



data_transform_aug, data_transform, trigger_transform, _, _ = supervisor.get_transforms(args)

if args.dataset == 'cifar10':

    num_classes = 10
    momentum = 0.9
    weight_decay = 1e-4
    milestones = [100, 150]
    epochs = 100
    learning_rate = 0.1
    test_freq = config.test_freq
    record = []

else:
    raise NotImplementedError()

if args.akl:
    epochs = 40
    test_freq = 0.125


if args.dataset in ['cifar10']:

    poison_set_dir = supervisor.get_dataset_dir(args)
    poisoned_set_img_path = os.path.join(poison_set_dir, 'imgs')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poisoned_set = tools.IMG_Dataset(data_path=poisoned_set_img_path,
                                         label_path=poisoned_set_label_path, transforms=data_transform if args.no_aug else  data_transform_aug)
    cleansed_set_indices_dir = supervisor.get_cleansed_set_indices_dir(args)
    print('load : %s' % cleansed_set_indices_dir)
    cleansed_set_indices = torch.load(cleansed_set_indices_dir)
else:
    raise NotImplementedError

poisoned_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))
cleansed_set_indices.sort()
poisoned_indices.sort()

tot_poison = len(poisoned_indices)
num_poison = 0

if tot_poison > 0:
    pt = 0
    for pid in cleansed_set_indices:
        while poisoned_indices[pt] < pid and pt + 1 < tot_poison: pt += 1
        if poisoned_indices[pt] == pid:
            num_poison += 1

print('remaining poison samples in cleansed set : ', num_poison)

cleansed_set = torch.utils.data.Subset(poisoned_set, cleansed_set_indices)
train_set = cleansed_set


if args.dataset in ['cifar10']:

    test_set_dir = supervisor.get_dataset_dir(args, split='test')
    test_set_img_path = os.path.join(test_set_dir, 'imgs')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_path=test_set_img_path,
                                 label_path=test_set_label_path, transforms=data_transform if args.no_aug else  data_transform_aug)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Poison Transform for Testing
    poison_transform = supervisor.get_poison_transform( target_class=config.target_class[args.dataset], trigger_transform=trigger_transform, is_normalized_input=True, args=args)

else:
    raise NotImplementedError


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

arch = supervisor.get_arch(args)

if args.poison_type == 'tact':
    source_classes = [config.args_default[args.dataset]['tact']['source_class']]
else:
    source_classes = None

model = arch(num_classes=num_classes)
model = nn.DataParallel(model)
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)


print(f'Train Set @ {poison_set_dir}')
print(f'Test Set @ {test_set_dir}')


for epoch in range(1,epochs+1):
    start_time = time.perf_counter()

    model.train()
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()  # train set batch
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('<Cleansed Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(), optimizer.param_groups[0]['lr'], elapsed_time))

    # Test
    if (epoch % max(int(epochs * test_freq), 1) == 0) and (epoch != epochs):
        if args.dataset == 'cifar10':
            # tools.test(model=model, test_loader=test_set_loader)
            ca, asr = tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes)
            torch.save(model.module.state_dict(), model_path)

            record.append((ca, asr))

        else:
            raise NotImplementedError()
# Final Test
if args.dataset == 'cifar10':
    ca, asr = tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, source_classes=source_classes, num_classes=num_classes)
    torch.save(model.module.state_dict(), model_path)

else:
    raise NotImplementedError()

torch.save(model.module.state_dict(), model_path)


file_name = f'a_{args.cleanser}_retrain.log'
file_path = os.path.join(model_dir, file_name)
with open(file_path, 'a') as file:
    file.write(f'[Time]: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n') 
    file.write(f'[Model]: {model_name}\n')
    file.write(f'[Cleanser]: {args.cleanser}\n')
    file.write(f'[Epochs]: {epochs}\n')
    file.write(f'[Trigger]: {args.trigger}\n')
    file.write(f'[Poison Type]: {args.poison_type}\n')
    file.write(f'[Poison Rate]: {args.poison_rate}\n')
    file.write(f'[Cover Rate]: {args.cover_rate}\n')
    file.write(f'[Alpha]: {args.alpha}\n')
    file.write(f'[Clean Accuracy]: {ca:.6f}\n')
    file.write(f'[Attack Success Rate]: {asr:.6f}\n')
    file.write(f'[Record]: {record}\n')
    file.write(f'\n')
    
