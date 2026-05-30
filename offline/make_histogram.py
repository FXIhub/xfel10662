import argparse

import h5py
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

LABELS = [
    'ref (2gtl)',
    '0.00 nm',
    '0.50 nm',
    '0.75 nm',
    '1.00 nm',
    '1.50 nm',
    '2.00 nm',
    'z-contact',
    'y-contact',
    'triplet',
    'ring',
    'cluster',
    'sphere',
]


def make_histogram_figure(iteration_info_path):
    with h5py.File(iteration_info_path) as f:
        c_d = f['iteration_0/most_likely_model_d'][()]

    H_c = np.bincount(c_d, minlength=12)
    bins = np.arange(len(H_c) + 1)

    fig, ax = plt.subplots(1, 1, height_ratios=[1])
    fig.set_size_inches(15, 5)
    fig.set_tight_layout(True)

    ax.bar(bins[:-1], H_c, width=1, align='edge', edgecolor='k', linewidth=1.0,
           color='lightcoral', label='classification by simulated classes')
    ax.set_xlim([bins.min(), bins.max()])
    ax.set_xlabel('class id')
    ax.set_ylabel('number')
    ax.legend()
    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xticks(bins[:-1])
    ax.set_xticklabels(LABELS)

    offset = transforms.ScaledTranslation(30 / 72, 0, fig.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="path", default="iteration_info.h5",
                        help="iteration_info.h5 path")
    parser.add_argument("--out", default="classification_histogram.png")
    args = parser.parse_args()

    fig = make_histogram_figure(args.path)
    fig.savefig(args.out)
    plt.show()
