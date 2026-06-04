import argparse

import numpy as np
import h5py

# NB: no module-level imports of the offline-local `common`/`constants` modules.
# This file is also exec'd as a library by DAMNIT's context.py (good_hits), where
# the offline/ dir is not on sys.path. `constants.PREFIX` is only needed by the
# CLI entry point, so it is imported lazily inside parse_args().

REQUIRED_DATASETS = {
    'class id':   '/entry_1/result_1/data',
    'size scale': '/entry_1/result_2/data',
    'hit_sigma':  '/entry_1/instrument_1/detector_1/score/hit_sigma',
}


def add_is_hit_cxi(cxi_file, hit_sigma_threshold = 10., size_min = 0.8, size_max = 1.2, good_classes = [0, 1, 2, 3, 4, 5, 6]):
    with h5py.File(cxi_file) as f:
        # Fail early and clearly if the upstream classification/sizing results
        # have not been written into the CXI yet (otherwise this surfaces as a
        # bare KeyError that's hard to interpret in the DAMNIT logs).
        missing = {label: path for label, path in REQUIRED_DATASETS.items()
                   if path not in f}
        if missing:
            details = '\n'.join(f'  - {label}: {path}' for label, path in missing.items())
            raise KeyError(
                f'{cxi_file}: missing dataset(s) required by add_is_hit_cxi; '
                f'has classification + sizing run for this run?\n{details}')

        cid = f[REQUIRED_DATASETS['class id']][()]
        scale_d = f[REQUIRED_DATASETS['size scale']][()]
        hit_sig = f[REQUIRED_DATASETS['hit_sigma']][()]

    # A frame is a "good hit" if it is in a good class, its fitted size scale is
    # within [size_min, size_max] along all three axes, and its hit_sigma clears
    # the threshold. Vectorised so this stays a few-millisecond operation.
    in_class = np.isin(cid, good_classes)
    in_size  = np.all((scale_d >= size_min) & (scale_d <= size_max), axis=1)
    above_sig = hit_sig > hit_sigma_threshold
    is_hit_g = (in_class & in_size & above_sig).astype(bool)

    # write to cxi file (overwrite if present so reprocessing is idempotent)
    is_hit_path = '/entry_1/instrument_1/detector_1/score/is_hit'
    with h5py.File(cxi_file, 'r+') as f:
        if is_hit_path in f:
            del f[is_hit_path]
        f[is_hit_path] = is_hit_g

    return is_hit_g

def parse_args():
    from constants import PREFIX
    p = argparse.ArgumentParser(description='Write is_hit based on size and classification to CXI file')
    p.add_argument('run', type=int, help='Run number')
    args = p.parse_args()

    args.output_file = f'{PREFIX}scratch/saved_hits/r{args.run:04d}_hits.cxi'
    return args

if __name__ == '__main__':
    args = parse_args()

    add_is_hit_cxi(args.output_file, hit_sigma_threshold = 10., size_min = 0.8, size_max = 1.2, good_classes = [0, 1, 2, 3, 4, 5, 6])
