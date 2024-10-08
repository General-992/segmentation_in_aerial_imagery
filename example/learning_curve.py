import argparse
import os.path as osp

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # NOQA
import pandas  # NOQA
import seaborn  # NOQA


def learning_curve(log_file):
    print('==> Plotting log file: %s' % log_file)

    if log_file is not str:
        df_list = []
        for log in log_file:
            df_piece = pandas.read_csv(log)
            df_list.append(df_piece)
        df = pandas.concat(df_list, ignore_index=True)
        log_file = log_file[-1]
    else:
        df = pandas.read_csv(log_file)
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    colors = seaborn.xkcd_palette(colors)

    plt.figure(figsize=(20, 6), dpi=300)

    row_min = df.min()
    row_max = df.max()

    # initialize DataFrame for train
    columns = [
        'epoch',
        'iteration',
        'train/loss',
        'train/acc',
        'train/acc_cls',
        'train/mean_iu',
        'train/fwavacc',
    ]
    df_train = df[columns].copy()
    if hasattr(df_train, 'rolling'):
        df_train = df_train.rolling(window=50).mean()
    else:
        df_train = pandas.rolling_mean(df_train, window=10)
    df_train = df_train.dropna()
    iter_per_epoch = df_train[df_train['epoch'] == 1]['iteration'].values[0]

    df_train['epoch_detail'] = df_train['iteration'] / iter_per_epoch

    # initialize DataFrame for val
    columns = [
        'epoch',
        'iteration',
        'valid/loss',
        'valid/acc',
        'valid/acc_cls',
        'valid/mean_iu',
        'valid/fwavacc',
    ]
    df_valid = df[columns].copy()
    df_valid = df_valid.dropna()
    df_valid['epoch_detail'] = df_valid['iteration'] / iter_per_epoch

    data_frames = {'train': df_train, 'valid': df_valid}

    n_row = 2
    n_col = 3
    for i, split in enumerate(['train', 'valid']):
        df_split = data_frames[split]

        # loss
        plt.subplot(n_row, n_col, i * n_col + 1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df_split['epoch_detail'], df_split['%s/loss' % split], '-',
                 markersize=1, color=colors[0], alpha=.5,
                 label='%s loss' % split)
        plt.xlim((0, row_max['epoch']))
        plt.ylim((min(row_min['train/loss'], row_min['valid/loss']),
                  max(row_max['train/loss'], row_max['valid/loss'])))
        plt.xlabel('epoch')
        plt.ylabel('%s loss' % split)

        # loss (log)
        plt.subplot(n_row, n_col, i * n_col + 2)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.semilogy(df_split['epoch_detail'], df_split['%s/loss' % split],
                     '-', markersize=1, color=colors[0], alpha=.5,
                     label='%s loss' % split)
        plt.xlim((0, row_max['epoch']))
        plt.ylim((min(row_min['train/loss'], row_min['valid/loss']),
                  max(row_max['train/loss'], row_max['valid/loss'])))
        plt.xlabel('epoch')
        plt.ylabel('%s loss (log)' % split)

        # lbl accuracy
        plt.subplot(n_row, n_col, i * n_col + 3)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df_split['epoch_detail'], df_split['%s/acc' % split],
                 '-', markersize=1, color=colors[1], alpha=.5,
                 label='%s accuracy' % split)
        plt.plot(df_split['epoch_detail'], df_split['%s/acc_cls' % split],
                 '-', markersize=1, color=colors[2], alpha=.5,
                 label='%s accuracy class' % split)
        plt.plot(df_split['epoch_detail'], df_split['%s/mean_iu' % split],
                 '-', markersize=1, color=colors[3], alpha=.5,
                 label='%s mean IU' % split)
        plt.plot(df_split['epoch_detail'], df_split['%s/fwavacc' % split],
                 '-', markersize=1, color=colors[4], alpha=.5,
                 label='%s fwav accuracy' % split)
        plt.legend()
        plt.xlim((0, row_max['epoch']))
        plt.ylim((0, 1))
        plt.xlabel('epoch')
        plt.ylabel('%s label accuracy' % split)

    out_file = osp.splitext(log_file)[0] + '.png'
    plt.savefig(out_file)
    print('==> Wrote figure to: %s' % out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', nargs='+', help='One or more log files to process')

    args = parser.parse_args()

    log_files = args.log_file if len(args.log_file) > 1 else args.log_file[0]

    learning_curve(log_files)


if __name__ == '__main__':
    main()
