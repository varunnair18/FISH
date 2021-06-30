from core import *
from asgd_torch_backend import *
from dawn_utils import net, tsv
import argparse
import os.path
from fisher import MultiFisherMask, MultiFullOneFisherMask, MultiRandomFisherMask

import random
import numpy as np
import torch
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--num_samples', type=int, default=1024)
parser.add_argument('--keep_ratio', type=float, default=0.005)
parser.add_argument('--mask_method', type=str, default="label-square", choices=["all_ones", "label-square", "random"])
parser.add_argument('--epochs', type=int, default=24)
parser.add_argument('--save_file', type=str, default="logs.tsv")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--merge_steps', default=10, type=int)
parser.add_argument('--recalculate_interval', default=10, type=int)
parser.add_argument('--split', default=2, type=int)
parser.add_argument('--diff_aggr_method', default="sum", type=str)
parser.add_argument('--lr', type=float, default=0.4)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def merge_models(models, merged_model, diff_aggr_method, same_classifier):
    # save the weight difference
    weight_diff = {}
    buffer_diff = {}

    for model in models:
        diff = {}
        for n, p in model.named_parameters():
            # the classifiers have different shapes
            if "classifier" in n and not same_classifier:
                continue
            pretrained_p = merged_model.state_dict()[n]
            diff[n] = p - pretrained_p

        if len(weight_diff) == 0:
            weight_diff.update(diff)
        else:
            for n, p in diff.items():
                weight_diff[n] += diff[n]

        b_diff = {}
        for n, p in model.named_buffers():
            # the classifiers have different shapes
            if "classifier" in n and not same_classifier:
                continue
            pretrained_p = merged_model.state_dict()[n]
            b_diff[n] = p - pretrained_p

        if len(buffer_diff) == 0:
            buffer_diff.update(b_diff)
        else:
            for n, p in b_diff.items():
                buffer_diff[n] += b_diff[n]

    if diff_aggr_method == 'mean':
        for n, p in weight_diff.items():
            weight_diff[n] = weight_diff[n] / len(models)

    for n, p in buffer_diff.items():
        buffer_diff[n] = buffer_diff[n] / len(models)

    weight_diff.update(buffer_diff)
    
    for n, p in merged_model.state_dict().items():
        if n not in weight_diff:
            continue

        diff_p = weight_diff[n]

        p.data.copy_(p.data + diff_p.data)
        # if n == "layer3_residual_res2_bn.running_var" or n == "layer3_residual_res2_bn.running_mean":
            # print(p.data)


def main():  
    args = parser.parse_args()

    set_seed(args)
    
    print('Downloading datasets')
    dataset = cifar10(args.data_dir)

    epochs = args.epochs
    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, args.lr, 0])
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    model = Network(net()).to(device).half()
    loss = x_ent_loss
    random_batch = lambda batch_size:  {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).cuda().half(), 
        'target': torch.LongTensor(np.random.randint(0,10,batch_size)).cuda()
    }
    print('Warming up cudnn on random inputs')
    for size in [batch_size, len(dataset['valid']['targets']) % batch_size]:
        warmup_cudnn(model, loss, random_batch(size))
    
    print('Starting timer')
    timer = Timer(synch=torch.cuda.synchronize)
    
    print('Preprocessing training data')
    transforms = [
        partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'), 
    ]
    train_set = list(zip(*preprocess(dataset['train'], [partial(pad, border=4)] + transforms).values()))
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(*preprocess(dataset['valid'], transforms).values()))
    print(f'Finished in {timer():.2} seconds')
    
    test_batches = DataLoader(test_set, batch_size, shuffle=False, drop_last=False)
    
    sample_type, grad_type = None, None

    if args.mask_method == "all_ones":
        MASK_CLASS = MultiFullOneFisherMask
    elif args.mask_method == "random":
        MASK_CLASS = MultiRandomFisherMask
    else:
        sample_type, grad_type = args.mask_method.split("-")
        MASK_CLASS = MultiFisherMask

    merged_model = model

    fisher_mask = MASK_CLASS(
        merged_model, 
        Transform(train_set, train_transforms), 
        args.num_samples, 
        args.keep_ratio, 
        sample_type, 
        grad_type,
        split=args.split
    )

    masks = fisher_mask.calculate_mask()

    merged_state = {MODEL: merged_model, LOSS: loss, "args": args}
    state_list = []
    train_batches_list = []
    for mask in masks:
        train_batches = DataLoader(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)

        model = deepcopy(merged_model)
        opts = [SGD(trainable_params(model).values(), {
            'lr': (lambda step: lr_schedule(step/len(train_batches))/batch_size), 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)})]

        state = {MODEL: model, LOSS: loss, OPTS: opts, "mask": mask, "args": args}

        state_list.append(state)
        train_batches_list.append(train_batches)

    logs = Table()

    step = 0

    for epoch in range(epochs):
        train_jobs = []
        for state, train_batches in zip(state_list, train_batches_list):
            train_jobs.append(train_epoch(state, timer, train_batches))

        for train_results in zip(*train_jobs):
            # extract models
            models = [state[MODEL] for state in state_list]

            # merge models
            merge_models(
                models, 
                merged_model, 
                args.diff_aggr_method, 
                True
            )

            if step % args.recalculate_interval == 0:
                # re-calulate masks
                masks = fisher_mask.calculate_mask()

                for state, mask in zip(state_list, masks):
                    state["mask"] = mask

            # re-assign weights
            for state in state_list:
                weight = merged_model.state_dict()
                for name, params in state[MODEL].state_dict().items():
                    params.data.copy_(weight[name].data)

            step += 1

        # compute the results, should be put in front of test_epoch
        log_dict = {f"train_{i}": epoch_stats(train_result) 
            for i, train_result in enumerate(train_results)}
        train_time = timer()
        
        valid_summary = test_epoch(state, timer, test_batches)

        valid_time = timer(include_in_total=False)
        
        log_dict['valid'] = union({'time': valid_time}, valid_summary)
        log_dict['total time'] = timer.total_time

        logs.append(union({'epoch': epoch+1}, log_dict))

    with open(os.path.join(os.path.expanduser(args.log_dir), args.save_file), 'w') as f:
        f.write(tsv(logs.log))        
        
main()