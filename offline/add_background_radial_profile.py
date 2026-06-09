"""
Compute the radial profile of the beamline background (data_white) in a hit CXI
file and write it back into the file.

The profile is the azimuthal average of data_white over the good pixels, binned
in detector-pixel units using the per-pixel xyz_map and x_pixel_size. It is
written to
    /entry_1/instrument_1/detector_1/background_radial_profile         (<B> at radius)
    /entry_1/instrument_1/detector_1/background_radial_profile_radius  (radius, pixels)
overwriting any previous values so reprocessing is idempotent.

Usage
-----
    python add_background_radial_profile.py 12                 # r0012_hits.cxi in scratch
    python add_background_radial_profile.py 12 --png prof.png  # also save a plot
    python add_background_radial_profile.py -c some.cxi        # explicit CXI file

NB: no module-level imports of the offline-local `constants` module, nor of
matplotlib. This file is also exec'd as a library by DAMNIT's context.py
(background_radial_profile), where the offline/ dir is not on sys.path; the CLI
needs constants only inside parse_args(), and the plot only with --png.
"""

import argparse

import h5py
import numpy as np

DET_PATH     = 'entry_1/instrument_1/detector_1'
PROFILE_PATH = f'{DET_PATH}/background_radial_profile'
RADIUS_PATH  = f'{DET_PATH}/background_radial_profile_radius'


class Radial():
    def __init__(self, mask, xyz=None, bin_size=1, center=None, rmax=None):
        # calculate r-values per pixel
        if xyz is None:
            inds = np.indices(mask.shape, dtype=float)
            for d in range(mask.ndim):
                if center is None:
                    c = mask.shape[d]//2
                else:
                    c = center[d]

                inds[d] -= c
        else:
            assert (xyz.shape[1:] == mask.shape)
            inds = xyz

        r = np.sqrt(np.sum(inds**2, axis=0))

        if rmax is not None:
            m = mask.copy()
            m[r>rmax] = False
        else:
            m = mask

        r = np.rint(r /  bin_size).astype(int)

        self.rbins_i = r[m]
        self.rbin_counts_r = np.bincount(self.rbins_i)
        self.rbin_counts_floor_r = np.clip(self.rbin_counts_r, 1, None)
        self.mask = m
        self.shape = self.rbin_counts_r.shape
        self.r = r

    def __call__(self, ar):
        rtot_r = np.bincount(self.rbins_i, weights=ar[self.mask])
        rav_r  = rtot_r / self.rbin_counts_floor_r
        return rav_r


def compute_background_profile(cxi_file, rmax_pixels=500):
    """Return (radius_pixels, B_r): the azimuthally averaged data_white."""
    with h5py.File(cxi_file) as f:
        B_i        = f[f'{DET_PATH}/data_white'][()]
        # xyz_map is (coordinate, module, y, x) with coordinate = (x, y, z) and
        # z = sample-detector distance. Use only the transverse (x, y) so the
        # radius is in the detector plane; including z would make r ~= distance
        # (>> rmax) and mask every pixel out.
        xyz        = f[f'{DET_PATH}/xyz_map'][:2]
        mask       = f[f'{DET_PATH}/mask'][()] == 0
        pixel_size = f[f'{DET_PATH}/x_pixel_size'][()]

    rav = Radial(mask, xyz, bin_size=pixel_size, rmax=rmax_pixels * pixel_size)
    B_r = rav(B_i)
    r = np.arange(B_r.shape[0])  # radius in pixel units (bin_size == pixel_size)
    return r, B_r


def add_background_radial_profile(cxi_file, rmax_pixels=500):
    """Compute the background radial profile and write it into the CXI file.

    Returns (radius_pixels, B_r)."""
    r, B_r = compute_background_profile(cxi_file, rmax_pixels)

    # overwrite if present so reprocessing is idempotent
    with h5py.File(cxi_file, 'r+') as f:
        for path, data in ((PROFILE_PATH, B_r), (RADIUS_PATH, r)):
            if path in f:
                del f[path]
            f[path] = data
    return r, B_r


def make_background_profile_figure(r, B_r, run='?'):
    """Plot the radial background profile; returns a matplotlib Figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, height_ratios=[1])
    fig.set_size_inches(8, 4)
    fig.set_tight_layout(True)

    ax.plot(r, B_r)

    ax.set_yscale('log')
    ax.set_xlabel('radius (pixels)')
    ax.set_ylabel('photons / pixel')
    ax.set_title(f'Radial profile of misses run {run}')

    ax.spines[['right', 'top']].set_visible(False)

    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description='Compute the background (data_white) radial profile and '
                    'write it into the CXI file.')
    p.add_argument('run', type=int, nargs='?',
                   help='run number -> r{run:04d}_hits.cxi in scratch/saved_hits')
    p.add_argument('-c', '--cxi', help='explicit CXI file path (overrides run)')
    p.add_argument('--rmax', type=int, default=500,
                   help='maximum radius in pixels (default 500)')
    p.add_argument('--png', help='also save a plot of the profile to this path')
    args = p.parse_args()

    if args.cxi:
        args.cxi_file = args.cxi
    elif args.run is not None:
        from constants import PREFIX
        args.cxi_file = f'{PREFIX}scratch/saved_hits/r{args.run:04d}_hits.cxi'
    else:
        p.error('give a run number or --cxi')
    return args


if __name__ == "__main__":
    args = parse_args()

    r, B_r = add_background_radial_profile(args.cxi_file, rmax_pixels=args.rmax)
    print(f'wrote {PROFILE_PATH} ({B_r.shape[0]} radial bins) to {args.cxi_file}; '
          f'median background = {np.median(B_r):.4g}')

    if args.png:
        run = args.run if args.run is not None else '?'
        fig = make_background_profile_figure(r, B_r, run)
        fig.savefig(args.png)
        print(f'saved plot to {args.png}')
