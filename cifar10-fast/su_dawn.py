from core import *
from su_torch_backend import *
from dawn_utils import net, tsv
import argparse
import os.path
from fisher import FisherMask, RandomFisherMask, FullOneFisherMask, calculate_the_importance_label

import random
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--num_samples', type=int, default=1024)
parser.add_argument('--keep_ratio', type=float, default=0.005)
parser.add_argument('--mask_method', type=str, default="label-square")
parser.add_argument('--epochs', type=int, default=24)
parser.add_argument('--save_file', type=str, default="logs.tsv")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--fix_mask', action="store_true", default=False)
parser.add_argument('--lr', type=float, default=0.4)
parser.add_argument('--model_name', type=str, default="resnet34")
parser.add_argument('--update_epoch_interval', type=int, default=1)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

     
def main():  
    args = parser.parse_args()

    set_seed(args)
    
    print('Downloading datasets')
    dataset = cifar10(args.data_dir)

    epochs = args.epochs

    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, args.lr, 0])
    
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    import torchvision as tv
    model_name = args.model_name

    if model_name == "resnet9":
        model = Network(net()).to(device)
    else:
        if model_name == "vit":
            from vit_pytorch import ViT
            model = ViT(
                image_size = 32,
                patch_size = 8,
                num_classes = 10,
                dim = 1024,
                depth = 6,
                heads = 4,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            ).to(device)
        elif model_name == "vit_deep":
            from vit_pytorch import ViT
            model = ViT(
                image_size = 32,
                patch_size = 8,
                num_classes = 10,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            ).to(device)
        elif model_name.startswith("resnet"):
            import resnet
            model_class = f"resnet.{model_name}"
            model = eval(model_class)(num_classes=10).to(device)
        else:
            model_class = f"tv.models.{model_name}"
            model = eval(model_class)(num_classes=10).to(device)

        def forward(self, inputs):
            outputs = {}
            outputs["logits"] = self._forward(inputs["input"])
            outputs["target"] = inputs["target"]

            return outputs

        def half(self):
            for n, p in self.named_parameters():
                # print(n)
                if "bn" not in n:
                    p.data = p.data.half()

            return self

        from types import MethodType

        setattr(model, '_forward', model.forward)
        setattr(model, 'forward', MethodType(forward, model))
        setattr(model, 'half', MethodType(half, model))
        # model._forward = model.forward
        # model.forward = forward
        # model.half = half
        # model = model.half()

    loss = x_ent_loss
    random_batch = lambda batch_size:  {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).cuda(), 
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
    
    train_batches = DataLoader(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    test_batches = DataLoader(test_set, batch_size, shuffle=False, drop_last=False)
    opts = [SGD(trainable_params(model).values(), {
        'lr': (lambda step: lr_schedule(step/len(train_batches))/batch_size), 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)})]

    sample_type, grad_type = None, None
    if args.mask_method == "random":
        MASK_CLASS = RandomFisherMask
    elif args.mask_method == "all_ones":
        MASK_CLASS = FullOneFisherMask
    else:
        sample_type, grad_type = args.mask_method.split("-")
        MASK_CLASS = FisherMask

    fisher_mask = MASK_CLASS(
        model, Transform(train_set, train_transforms), args.num_samples, args.keep_ratio, sample_type, grad_type
    )
    
    logs = Table()
    state = {MODEL: model, LOSS: loss, OPTS: opts, "fisher_mask": fisher_mask, "args": args}

    for epoch in range(epochs):
        state["epoch"] = epoch + 1
        logs.append(union({'epoch': epoch+1}, train_epoch(state, timer, train_batches, test_batches)))

    # def tsv(logs):
    #     data = [(output['epoch'], output['total time']/3600, output['valid']['acc']*100) for output in logs]
    #     return '\n'.join(['epoch\thours\ttop1Accuracy']+[f'{epoch}\t{hours:.8f}\t{acc:.2f}' for (epoch, hours, acc) in data])

    with open(os.path.join(os.path.expanduser(args.log_dir), args.save_file), 'w') as f:
        f.write(tsv(logs.log))        
        
main()