import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from utils import supervisor

import config

# 
def get_features(data_loader, model):
    
    label_list = []
    preds_list = []
    feats = []
    gt_confidence = []
    loss_vals = []

    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    model.eval()

    with torch.no_grad():

        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):

            ins_data, ins_target = ins_data.cuda(), ins_target.cuda()
            output, x_features = model(ins_data, return_hidden=True)

            loss = criterion_no_reduction(output, ins_target).cpu().numpy()

            preds = torch.argmax(output, dim=1).cpu().numpy()
            prob = torch.softmax(output, dim=1).cpu().numpy()
            this_batch_size = len(ins_target)

            for bid in range(this_batch_size):
                gt = ins_target[bid].cpu().item()
                feats.append(x_features[bid].cpu().numpy())
                label_list.append(gt)
                preds_list.append(preds[bid])
                gt_confidence.append(prob[bid][gt])
                loss_vals.append(loss[bid])
    return feats, label_list, preds_list, gt_confidence, loss_vals

def identify_poison_samples_simplified(inspection_set, clean_indices, model, num_classes):
    from scipy.stats import multivariate_normal

    num_samples = len(inspection_set)
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False)

    model.eval()
    feats_inspection, class_labels_inspection, preds_inspection, \
    gt_confidence_inspection, loss_vals = get_features(inspection_split_loader, model)

    feats_inspection = np.array(feats_inspection)
    class_labels_inspection = np.array(class_labels_inspection)

    class_indices = [[] for _ in range(num_classes)]
    class_indices_in_clean_chunklet = [[] for _ in range(num_classes)]

    for i in range(num_samples):
        gt = int(class_labels_inspection[i])
        class_indices[gt].append(i)

    for i in clean_indices:
        gt = int(class_labels_inspection[i])
        class_indices_in_clean_chunklet[gt].append(i)

    for i in range(num_classes):
        class_indices[i].sort()
        class_indices_in_clean_chunklet[i].sort()

        if len(class_indices[i]) < 2:
            raise Exception('dataset is too small for class %d' % i)

        if len(class_indices_in_clean_chunklet[i]) < 2:
            raise Exception('clean chunklet is too small for class %d' % i)
        
    threshold = 2
    suspicious_indices = []
    class_likelihood_ratio = []

    for target_class in range(num_classes):

        num_samples_within_class = len(class_indices[target_class])
        clean_chunklet_size = len(class_indices_in_clean_chunklet[target_class])
        clean_chunklet_indices_within_class = []
        pt = 0
        for i in range(num_samples_within_class):
            if pt == clean_chunklet_size:
                break
            if class_indices[target_class][i] < class_indices_in_clean_chunklet[target_class][pt]:
                continue
            else:
                clean_chunklet_indices_within_class.append(i)
                pt += 1

        temp_feats = torch.FloatTensor(
            feats_inspection[class_indices[target_class]]).cuda()


        # reduce dimensionality
        U, S, V = torch.pca_lowrank(temp_feats, q=2)
        projected_feats = torch.matmul(temp_feats, V[:, :2]).cpu()

        # isolate samples via the confused inference model
        isolated_indices_global = []
        isolated_indices_local = []
        other_indices_local = []
        labels = []
        for pt, i in enumerate(class_indices[target_class]):
            if preds_inspection[i] == target_class:
                isolated_indices_global.append(i)
                isolated_indices_local.append(pt)
                labels.append(1) # suspected as positive
            else:
                other_indices_local.append(pt)
                labels.append(0)

        projected_feats_isolated = projected_feats[isolated_indices_local]
        projected_feats_other = projected_feats[other_indices_local]

        num_isolated = projected_feats_isolated.shape[0]

        if (num_isolated >= 2) and (num_isolated <= num_samples_within_class - 2):

            mu = np.zeros((2,2))
            covariance = np.zeros((2,2,2))

            mu[0] = projected_feats_other.mean(axis=0)
            covariance[0] = np.cov(projected_feats_other.T)
            mu[1] = projected_feats_isolated.mean(axis=0)
            covariance[1] = np.cov(projected_feats_isolated.T)

            # avoid singularity
            covariance += 0.001

            # likelihood ratio test
            single_cluster_likelihood = 0
            two_clusters_likelihood = 0
            for i in range(num_samples_within_class):
                single_cluster_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[0],
                                                                        cov=covariance[0],
                                                                        allow_singular=True).sum()
                two_clusters_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[labels[i]],
                                                                      cov=covariance[labels[i]], allow_singular=True).sum()

            likelihood_ratio = np.exp( (two_clusters_likelihood - single_cluster_likelihood) / num_samples_within_class )

        else:

            likelihood_ratio = 1

        class_likelihood_ratio.append(likelihood_ratio)

    max_ratio = np.array(class_likelihood_ratio).max()

    for target_class in range(num_classes):
        likelihood_ratio = class_likelihood_ratio[target_class]

        if likelihood_ratio == max_ratio and likelihood_ratio > 1.5:

            for i in class_indices[target_class]:
                if preds_inspection[i] == target_class:
                    suspicious_indices.append(i)

        elif likelihood_ratio > threshold:

            for i in class_indices[target_class]:
                if preds_inspection[i] == target_class:
                    suspicious_indices.append(i)

        else:
            pass

    return suspicious_indices



#
def pretrain(args, arch, num_classes, weight_decay, pretrain_epochs, distilled_set_loader, criterion, inspection_set_dir, confusion_iter, lr, load = True, dataset_name=None):
    model = arch(num_classes = num_classes)

    if confusion_iter != 0 and load:
        ckpt_path = os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter-1, args.seed))
        model.load_state_dict( torch.load(ckpt_path) )

    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr,  momentum=0.9, weight_decay=weight_decay)

    for epoch in tqdm(range(1, pretrain_epochs + 1)):
        model.train()

        for batch_idx, (data, target) in enumerate(distilled_set_loader):
            optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()  # train set batch
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    base_ckpt = model.module.state_dict()
    torch.save(base_ckpt, os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed)))
    print('save : ', os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed)))

    return model

def confusion_train(args, params, inspection_set, debug_packet, distilled_set_loader, clean_set_loader, confusion_iter, arch, num_classes, inspection_set_dir, weight_decay, criterion_no_reduction, momentum, lamb, freq, lr, batch_factor, distillation_iters, dataset_name = None):

    base_model = params['arch'](num_classes = num_classes)
    model_path, _, _ = supervisor.get_model_path(args, cleanse='ct')
    
    base_model.load_state_dict(torch.load(model_path))
    base_model = nn.DataParallel(base_model)
    base_model = base_model.cuda()
    base_model.eval()


    ######### Distillation Step ################

    model = arch(num_classes = num_classes)
    model.load_state_dict(
                torch.load(os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed)))
    )
    model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                momentum=momentum)

    distilled_set_iters = iter(distilled_set_loader)
    clean_set_iters = iter(clean_set_loader)


    rounder = 0

    for batch_idx in tqdm(range(distillation_iters)):

        try:
            data_shift, target_shift = next(clean_set_iters)
        except Exception as e:
            clean_set_iters = iter(clean_set_loader)
            data_shift, target_shift = next(clean_set_iters)
        data_shift, target_shift = data_shift.cuda(), target_shift.cuda()

        if dataset_name == 'cifar10':
            with torch.no_grad():
                preds = torch.argmax(base_model(data_shift), dim=1).detach()
                if (rounder + batch_idx) % num_classes == 0:
                    rounder += 1
                next_target = (preds + rounder + batch_idx) % num_classes
                target_confusion = next_target
        else:
            raise NotImplementedError()

        model.train()

        if batch_idx % batch_factor == 0:

            try:
                data, target = next(distilled_set_iters)
            except Exception as e:
                distilled_set_iters = iter(distilled_set_loader)
                data, target = next(distilled_set_iters)

            data, target = data.cuda(), target.cuda()
            data_mix = torch.cat([data_shift, data], dim=0)
            target_mix = torch.cat([target_confusion, target], dim=0)
            boundary = data_shift.shape[0]

            output_mix = model(data_mix)
            loss_mix = criterion_no_reduction(output_mix, target_mix)

            loss_inspection_batch_all = loss_mix[boundary:]
            loss_confusion_batch_all = loss_mix[:boundary]
            loss_confusion_batch = loss_confusion_batch_all.mean()
            target_inspection_batch_all = target_mix[boundary:]
            inspection_batch_size = len(loss_inspection_batch_all)
            loss_inspection_batch = 0
            normalizer = 0
            for i in range(inspection_batch_size):
                gt = int(target_inspection_batch_all[i].item())
                loss_inspection_batch += (loss_inspection_batch_all[i] / freq[gt])
                normalizer += (1 / freq[gt])
            loss_inspection_batch = loss_inspection_batch / normalizer

            weighted_loss = (loss_confusion_batch * (lamb-1) + loss_inspection_batch) / lamb

            loss_confusion_batch = loss_confusion_batch.item()
            loss_inspection_batch = loss_inspection_batch.item()
        else:
            output = model(data_shift)
            weighted_loss = loss_confusion_batch = criterion_no_reduction(output, target_confusion).mean()
            loss_confusion_batch = loss_confusion_batch.item()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

    torch.save( model.module.state_dict(),
               os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (confusion_iter, args.seed)) )
    print('save : ', os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (confusion_iter, args.seed)))

    return model


def distill(args, params, inspection_set, n_iter, criterion_no_reduction, dataset_name = None, final_budget = None, class_wise = False, custom_arch=None):

    kwargs = params['kwargs']
    inspection_set_dir = params['inspection_set_dir']
    num_classes = params['num_classes']
    num_samples = len(inspection_set)
    arch = params['arch']
    distillation_ratio = params['distillation_ratio']
    num_confusion_iter = len(distillation_ratio) + 1

    if custom_arch is not None:
        arch = custom_arch

    model = arch(num_classes=num_classes)
    ckpt = torch.load(os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (n_iter, args.seed)))
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model)
    model = model.cuda()
    inspection_set_loader = torch.utils.data.DataLoader(inspection_set, batch_size=256, shuffle=False)

    loss_array = []
    correct_instances = []
    gts = []
    model.eval()
    st = 0
    with torch.no_grad():

        for data, target in tqdm(inspection_set_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)

            if dataset_name == 'cifar10':
                preds = torch.argmax(output, dim=1)
            else :
                raise NotImplementedError()

            batch_loss = criterion_no_reduction(output, target)

            this_batch_size = len(target)

            for i in range(this_batch_size):
                loss_array.append(batch_loss[i].item())
                gts.append(int(target[i].item()))
                if dataset_name != 'cifar10':
                    if preds[i] == target[i]:
                        correct_instances.append(st + i)

            st += this_batch_size

    loss_array = np.array(loss_array)
    sorted_indices = np.argsort(loss_array)

    top_indices_each_class = [[] for _ in range(num_classes)]
    for t in sorted_indices:
        gt = gts[t]
        top_indices_each_class[gt].append(t)

    if n_iter < num_confusion_iter - 1:

        if distillation_ratio[n_iter] is None:
            distilled_samples_indices = head = correct_instances
        else:
            num_expected = int(distillation_ratio[n_iter] * num_samples)
            head = sorted_indices[:num_expected]
            head = list(head)
            distilled_samples_indices = head

        if n_iter < num_confusion_iter - 2: rate_factor = 50
        else: rate_factor = 100
        class_dist = np.zeros(num_classes, dtype=int)
        
        for i in distilled_samples_indices:
            gt = gts[i]
            class_dist[gt] += 1

        for i in range(num_classes):
            minimal_sample_num = len(top_indices_each_class[i]) // rate_factor
            if class_dist[i] < minimal_sample_num:
                for k in range(class_dist[i], minimal_sample_num):
                    distilled_samples_indices.append(top_indices_each_class[i][k])

    else:
        if final_budget is not None:
            head = sorted_indices[:final_budget]
            head = list(head)
            distilled_samples_indices = head
        else:
            distilled_samples_indices = head = correct_instances

    distilled_samples_indices.sort()


    median_sample_rate = params['median_sample_rate']
    median_sample_indices = []
    sorted_indices_each_class = [[] for _ in range(num_classes)]
    for temp_id in sorted_indices:
        gt = gts[temp_id]
        sorted_indices_each_class[gt].append(temp_id)

    for i in range(num_classes):
        num_class_i = len(sorted_indices_each_class[i])
        st = int(num_class_i / 2 - num_class_i * median_sample_rate / 2)
        ed = int(num_class_i / 2 + num_class_i * median_sample_rate / 2)
        for temp_id in range(st, ed):
            median_sample_indices.append(sorted_indices_each_class[i][temp_id])

    if class_wise:
        return distilled_samples_indices, median_sample_indices, top_indices_each_class
    else:
        return distilled_samples_indices, median_sample_indices
    
def iterative_poison_distillation(inspection_set, clean_set, params, args, debug_packet=None, start_iter=0):

    inspection_set_dir = params['inspection_set_dir']
    num_classes = params['num_classes']
    pretrain_epochs = params['pretrain_epochs']
    weight_decay = params['weight_decay']
    arch = params['arch']
    distillation_ratio = params['distillation_ratio']
    momentums = params['momentums']
    lambs = params['lambs']
    lrs = params['lrs']
    batch_factor = params['batch_factors']

    clean_set_loader = torch.utils.data.DataLoader(
        clean_set, batch_size=params['batch_size'],
        shuffle=True)

    distilled_samples_indices, median_sample_indices = None, None
    num_confusion_iter = len(distillation_ratio) + 1
    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()

    distilled_set = inspection_set


    for confusion_iter in range(start_iter, num_confusion_iter):

        size_of_distilled_set = len(distilled_set)
        print('<Round-%d> Size_of_distillation_set = ' % confusion_iter, size_of_distilled_set)
        
        nums_of_each_class = np.zeros(num_classes)
        for i in range(size_of_distilled_set):
            _, gt = distilled_set[i]
            gt = gt.item()
            nums_of_each_class[gt] += 1
        freq_of_each_class = nums_of_each_class / size_of_distilled_set
        freq_of_each_class = np.sqrt(freq_of_each_class + 0.001)

        if confusion_iter < 2: # lr=0.01 for round 0,1,2
            pretrain_epochs = 100
            pretrain_lr = 0.01
            distillation_iters = 6000
        elif confusion_iter < 3: # lr=0.01 for round 0,1,2
            pretrain_epochs = 40
            pretrain_lr = 0.01
            distillation_iters = 6000
        elif confusion_iter < 4:
            pretrain_epochs = 40
            pretrain_lr = 0.01
            distillation_iters = 6000
        elif confusion_iter < 5:
            pretrain_epochs = 40
            pretrain_lr = 0.01
            distillation_iters = 2000
        else:
            pretrain_epochs = 40
            pretrain_lr = 0.01
            distillation_iters = 2000


        lr = lrs[confusion_iter]

        if confusion_iter == num_confusion_iter - 1:
            freq_of_each_class[:] = 1

        if confusion_iter != num_confusion_iter - 1:
            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=params['batch_size'], shuffle=True)
        else:
            distilled_set_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([distilled_set, clean_set]),
                batch_size=params['batch_size'], shuffle=True)

        # pretrain base model
        pretrain(args, arch, num_classes, weight_decay, pretrain_epochs,
                                        distilled_set_loader, criterion, inspection_set_dir, confusion_iter, pretrain_lr)

        distilled_set_loader = torch.utils.data.DataLoader(
            distilled_set,
            batch_size=params['batch_size'], shuffle=True)

        # confusion_training
        model = confusion_train(args, params, inspection_set, debug_packet, distilled_set_loader, clean_set_loader, confusion_iter, arch,
                                   num_classes, inspection_set_dir, weight_decay, criterion_no_reduction,
                                   momentums[confusion_iter], lambs[confusion_iter],
                                   freq_of_each_class, lr, batch_factor[confusion_iter], distillation_iters)

        # distill the inspected set according to the loss values
        distilled_samples_indices, median_sample_indices = distill(args, params, inspection_set, confusion_iter, criterion_no_reduction)

        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)

    return distilled_samples_indices, median_sample_indices, model

def cleanser(poisoned_set, clean_set, model, num_classes, args):

    params = config.get_params(args)
    inspection_set = poisoned_set
    
    _, median_sample_indices, model = iterative_poison_distillation(inspection_set, clean_set, params, args, start_iter=0)

    suspicious_indices = identify_poison_samples_simplified(inspection_set, median_sample_indices, model, num_classes=params['num_classes'])

    return suspicious_indices