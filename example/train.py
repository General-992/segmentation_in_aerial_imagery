import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess

import torch
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../torchconvs')))

import torchconvs
import scripts
def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()

    # Check for uncommitted changes
    status_cmd = 'git status --porcelain'
    status = subprocess.check_output(shlex.split(status_cmd)).strip()
    if status:
        ret += '-dirty'
    return ret
def none_or_int(value):
    if value.lower() == 'none':
        return None
    return int(value)

cuda = torch.cuda.is_available()
here = osp.dirname(osp.abspath(__file__))


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--resume', help='checkpoint path')
    parser.add_argument(
        '--max-epoch', type=int, default=100, help='max epoch'
    )
    parser.add_argument(
        '--patch-size', type= none_or_int, default=256, help="Patch size for patchifying. Use 'None' to disable patchifying.",
    )
    parser.add_argument(
        '--max-lr', type=float, default=1.0e-3, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0001, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='momentum',
    )
    parser.add_argument(
        '--batch-size', type=int, default=16, help='batch size',
    )
    parser.add_argument(
        '--optim', type=str, default='adamw', help='optimizer',
    )
    parser.add_argument(
        '--data-path', type=str, help='specify path to dataset',
    )

    args = parser.parse_args()
    args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', f'{args.model}_{now.strftime("%Y_%m_%d.%H_%M")}')

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cuda = torch.cuda.is_available()
    print(f'Cuda available: {cuda}')
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    if args.data_path is None:
        root = osp.expanduser('~/flair_dataset')
    else:
        root = args.data_path
    file_list = os.listdir(root)
    required = {'img', 'msk', 'train.txt', 'val.txt'}
    if not required.issubset(set(file_list)):
        raise Exception('Dataset repository setup is incorrect')

    kwargs = {'num_workers': 12, 'pin_memory': True, 'prefetch_factor': 2} if cuda else {}
    ## TODO set configurable train-val batchsize
    train_loader = torch.utils.data.DataLoader(
        torchconvs.datasets.FLAIRSegBase(root, split='train', patch_size=args.patch_size, transform=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)


    val_loader = torch.utils.data.DataLoader(
        torchconvs.datasets.FLAIRSegBase(root, split='val', patch_size=args.patch_size, transform=True),
        batch_size=24, shuffle=False, **kwargs)


    n_class = train_loader.dataset.class_names.shape[0]
    print(f'Number of semantic classes: {n_class}')
    # 2. model

    model = scripts.utils.model_select(args.model, n_class)

    start_epoch = 0
    start_iteration = 0
    if args.resume:
        print(f'Resuming from: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']

    if cuda:
        model = model.cuda()

    # 3. optimizer

    ## TODO: implement configurable lr
    # optim = torch.optim.SGD(
    #     params=model.parameters(),
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)

    # the range of lr for a Adamw is 1e-3 while for SGD is 1e-10 x_x
    if args.optim.lower() == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception(f'Unsupported optimizer: {args.optim}')

    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchconvs.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_epoch=args.max_epoch,
        interval_validate=None,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
