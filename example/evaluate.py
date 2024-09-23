import argparse
import os
import os.path as osp

import fcn
import numpy as np
import skimage.io
import torch

import torchconvs
import scripts
import tqdm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    root = osp.expanduser('~/datasets/flair_dataset')
    val_loader = torch.utils.data.DataLoader(
        torchconvs.datasets.FLAIRSegBase(
            root, split='val', transform=False),
        batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True)

    # # Patch-Based training and Full image Testing
    # test_dataset = torchconvs.datasets.FLAIRSegBase(root='data', split='test', transform=True,
    #                                                 patch_size=256, test=True, inference_mode='original')
    #
    # # Patch-Based training and Tile-Based image Testing
    # test_dataset = torchconvs.datasets.FLAIRSegBase(root='data', split='test', transform=True,
    #                                                 patch_size=256, test=True, inference_mode='tiles')
    #
    # # Full image training and Full image Testing
    # test_dataset = torchconvs.datasets.FLAIRSegBase(root='data', split='test', transform=True,
    #                                                 patch_size=None, test=True, inference_mode='original')

    n_class = len(val_loader.dataset.class_names)
    model_data = torch.load(model_file)
    print(model_data['arch'])
    model = scripts.utils.model_select(model_data['arch'], n_class)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))

    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating %s' % model.__class__.__name__)
    visualizations = []
    metrics = []
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            score = model(data)

        # Safely detach from the computation graph and move to CPU
        imgs = data.detach().cpu()
        lbl_pred = score.detach().max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.detach().cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            # reconstruct the original images
            img, lt = val_loader.dataset.untransform(img, lt)

            acc, acc_cls, mean_iu, fwavacc = scripts.utils.label_accuracy_score(
                label_trues=lt, label_preds=lp, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))

            if len(visualizations) < 6:
                viz = fcn.utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    label_names=val_loader.dataset.class_names)
                visualizations.append(viz)
    metrics = np.mean(metrics, axis=0)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

    viz = fcn.utils.get_tile_image(visualizations)
    skimage.io.imsave('viz_evaluate.png', viz)


if __name__ == '__main__':
    main()
