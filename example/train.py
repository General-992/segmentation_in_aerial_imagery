import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess

import torch
import yaml

import torchconvs


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


cuda = torch.cuda.is_available()
here = osp.dirname(osp.abspath(__file__))


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('-model', type=str, help='model name')
    parser.add_argument('--resume', help='checkpoint path')
    parser.add_argument(
        '--max-epoch', type=int, default=100, help='max epoch'
    )
    parser.add_argument(
        '--max-lr', type=float, default=1.0e-3, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--batch-size', type=int, default=24, help='batch size',
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

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    root = osp.expanduser('~/datasets/flair_dataset')
    kwargs = {'num_workers': 4, 'pin_memory': True, 'prefetch_factor': 2} if cuda else {}

    train_loader = torch.utils.data.DataLoader(
        torchconvs.datasets.FLAIRSegBase(root, split='train', transform=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        torchconvs.datasets.FLAIRSegBase(root, split='val', transform=True),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # 2. model
    if args.model.lower() == 'unetplusplus':
        # num of trainable params = 48.986.615
        print('Start training Unet')
        model = torchconvs.models.UnetPlusPlus(n_class=7)
    elif args.model.lower() == 'deeplab':
        #  num of trainable params = 39.758.247
        print('Start training Deeplab')
        model = torchconvs.models.DeepLabV3Plus.Deeplabv3plus_resnet(n_class=7)
    elif args.model.lower() == 'segnet':
        print('Start training Segnet')
        # num of trainable params = 12.932.295
        model = torchconvs.models.SegNet(num_classes=7)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    else:
        raise Exception('Unknown model')

    start_epoch = 0
    start_iteration = 0
    if args.resume:
        print(f'Resuming from: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    # else:
        # vgg16 = torchfcn.models.VGG16(pretrained=True)
        # model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    # optim = torch.optim.SGD(
    #     params=model.parameters(),
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)

    # the range of lr for a Adamw is 1e-3 while for SGD is 1e-10 x_x
    optim = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
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
