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

import json


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    root = osp.expanduser('~/datasets/flair_dataset')

    # Specify the file path
    file_path = osp.join(root, 'flair-1_metadata_aerial.json')
    with open(file_path, 'r') as file:
        data = json.load(file)


    val_loader = torch.utils.data.DataLoader(
        torchconvs.datasets.FLAIRSegMeta(
            root=root,metadata=data, split='val', transform=False),
        batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True)


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
        # For earlier torch versions
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating Metadata%s' % model.__class__.__name__)

    month_metrics = {str(i): [] for i in range(1, 13)}
    for batch_idx, (data, target, month, camera) in tqdm.tqdm(enumerate(val_loader),
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

        for img, lt, lp, m in zip(imgs, lbl_true, lbl_pred, month):
            # Reconstruct the original images
            img, lt = val_loader.dataset.untransform(img, lt)

            # Calculate metrics for the current image
            acc, acc_cls, mean_iu, fwavacc = scripts.utils.label_accuracy_score(
                label_trues=lt, label_preds=lp, n_class=n_class)

            # Append the metrics to the correct month entry
            month_str = str(int(m))  # Ensure the month is a string (e.g., '1', '2', ..., '12')
            if month_str in month_metrics:
                month_metrics[month_str].append((acc, acc_cls, mean_iu, fwavacc))
    ## TODO: has not checked this part
    for month, metrics in month_metrics.items():
        if metrics:
            metrics = np.mean(metrics, axis=0)
            metrics *= 100  # Convert to percentages
            print(f"Month: {month}")
            print('''\
            Accuracy: {0}
            Accuracy Class: {1}
            Mean IU: {2}
            FWAV Accuracy: {3}'''.format(*metrics))
    scripts.utils.plot_metrics_per_month(month_metrics)





if __name__ == '__main__':
    main()
