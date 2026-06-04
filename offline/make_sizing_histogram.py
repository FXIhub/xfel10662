import argparse

import h5py
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

def make_sizing_histogram(cxi_file, run='?'):
    with h5py.File(cxi_file) as f:
        frames = f['entry_1/result_2/frames'][()]
        # scale_index_d = f['entry_1/result_2/scale_index'][frames]
        scale_d = f['entry_1/result_2/data'][()][frames]
        scale_unique = f['entry_1/result_2/scale_unique'][()]

    # get grid (assume scale_x == scale_y)
    # also assume grid like steps
    scales = scale_unique.T

    if not np.allclose(scales[0], scales[1]):
        print('Warning scale_x != scale_y breaking assumption')

    scale_x = np.unique(scales[0])
    scale_z = np.unique(scales[2])

    dx = scale_x[1] - scale_x[0]
    dz = scale_z[1] - scale_z[0]

    Nx = len(scale_x)
    Nz = len(scale_z)

    bins_x = dx * np.arange(len(scale_x)+1) + scale_x[0] - dx/2
    bins_z = dz * np.arange(len(scale_z)+1) + scale_z[0] - dz/2

    hist, _, _ = np.histogram2d(scale_d[:, 0], scale_d[:, 2], (bins_x, bins_z))

    fig, ax = plt.subplots(1, 1, height_ratios=[1])
    fig.set_size_inches(8, 8)
    fig.set_tight_layout(True)

    ax.imshow(hist, extent=[bins_x[0], bins_x[-2], bins_z[0], bins_z[-2]], origin='lower')

    ax.set_xlabel('scale z')
    ax.set_ylabel('scale x,y')
    ax.set_title(f'Ery 0.75 nm size histogram run {run}')

    ax.spines[['right', 'top']].set_visible(False)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="path", default="r001_hits.cxi",
                        help="cxi file path")
    parser.add_argument("--out", default="sizing_histogram.png")
    args = parser.parse_args()

    fig = make_sizing_histogram(args.path)
    fig.savefig(args.out)
    plt.show()
