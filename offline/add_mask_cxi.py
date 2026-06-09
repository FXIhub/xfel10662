"""
Overwrite the per-pixel mask in hit CXI files from a good-pixel mask file.

The mask file holds a boolean good-pixel map (True = good, False = bad) shaped
like the detector (NMODULES, ss, fs). It is written into each target CXI file at
    /entry_1/instrument_1/detector_1/mask
in CXI format, where 0 = good and CXI_PIXEL_IS_BAD is set on bad pixels (the
inverse convention of the input). The frame data itself is never touched.

Selecting which CXI files to update
-----------------------------------
    python add_mask_cxi.py mask.h5 -r 12 13 17     # r{run}_hits.cxi in --dir
    python add_mask_cxi.py mask.h5 -c some.cxi      # one explicit CXI file
    python add_mask_cxi.py mask.h5                  # every r*_hits.cxi in --dir

The mask dataset is auto-detected (looking for /data, /entry_1/good_pixels,
/mask, /good_pixels) or can be given as path/to/file.h5:/dataset.
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

from constants import PREFIX

SAVED_HITS_DIR = f'{PREFIX}scratch/saved_hits'

CXI_MASK_PATH = 'entry_1/instrument_1/detector_1/mask'

# CXI mask bits (cxi.h); matches make_cxi_file.py
CXI_PIXEL_IS_BAD = 0x00000080

# datasets tried, in order, when none is given explicitly
CANDIDATE_MASK_DSETS = ('data', 'entry_1/good_pixels', 'mask', 'good_pixels')


def load_good_pixels(spec):
    """Load a boolean good-pixel map (True = good) from FILE or FILE:/dataset."""
    if ':' in spec and not os.path.exists(spec):
        path, _, dset = spec.rpartition(':')
        dsets = [dset.lstrip('/')]
    else:
        path = spec
        dsets = CANDIDATE_MASK_DSETS

    with h5py.File(path, 'r') as f:
        for dset in dsets:
            if dset in f:
                arr = f[dset][()]
                print(f'read good-pixel mask from {path}:/{dset} shape {arr.shape}')
                return arr.astype(bool)
        raise SystemExit(
            f'no mask dataset found in {path} (tried {", ".join(dsets)}); '
            'pass it explicitly as FILE:/dataset')


def write_mask(cxi_file, good):
    """Overwrite the CXI mask in cxi_file from the good-pixel map `good`."""
    cxi_mask = np.where(good, 0, CXI_PIXEL_IS_BAD).astype(np.uint32)
    with h5py.File(cxi_file, 'r+') as f:
        if CXI_MASK_PATH not in f:
            print(f'warning: {cxi_file} has no {CXI_MASK_PATH}; skipping')
            return False
        dset = f[CXI_MASK_PATH]
        if dset.shape != cxi_mask.shape:
            print(f'warning: {cxi_file} mask shape {dset.shape} != '
                  f'{cxi_mask.shape}; skipping')
            return False
        dset[...] = cxi_mask
    nbad = int((~good).sum())
    print(f'updated {cxi_file}: {nbad} bad / {good.size} pixels')
    return True


def main():
    p = argparse.ArgumentParser(
        description='Overwrite the mask in cxi files from a good-pixel mask '
                    '(True = good pixel, False = bad pixel).')
    p.add_argument('mask', help='good-pixel mask file, optionally FILE:/dataset')
    p.add_argument('-r', '--runs', nargs='+', metavar='RUN',
                   help='run list -> r{run:04d}_hits.cxi in --dir, or "all" '
                        'for every r*_hits.cxi in --dir')
    p.add_argument('-c', '--cxi', help='update this CXI file explicitly')
    p.add_argument('-d', '--dir', default=SAVED_HITS_DIR,
                   help=f'directory of r####_hits.cxi files (default {SAVED_HITS_DIR})')
    args = p.parse_args()

    good = load_good_pixels(args.mask)

    if args.cxi:
        cxi_files = [args.cxi]
    elif args.runs and 'all' not in args.runs:
        cxi_files = [os.path.join(args.dir, f'r{int(run):04d}_hits.cxi')
                     for run in args.runs]
    else:
        cxi_files = sorted(glob.glob(os.path.join(args.dir, 'r*_hits.cxi')))

    if not cxi_files:
        sys.exit(f'no CXI files to update (dir {args.dir})')

    n = 0
    for cxi_file in cxi_files:
        if not os.path.exists(cxi_file):
            print(f'warning: {cxi_file} does not exist; skipping')
            continue
        n += write_mask(cxi_file, good)
    print(f'done: updated {n}/{len(cxi_files)} files')


if __name__ == '__main__':
    main()
