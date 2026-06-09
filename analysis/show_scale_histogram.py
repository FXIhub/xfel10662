"""Sizing-scale histograms combined over a set of runs.

Two figures are produced:

  1. A 1D histogram of the average scale factor (scale_xy + scale_z)/2.
     For each frame the sizing fit reports a scale [x, y, z] picked from a
     discrete grid (linspace(0.6, 1.4, 17), see offline/config_sizing.py). The
     grid edges are the limits of the search range, so a frame that lands on the
     lowest or highest grid value in either axis is "pinned" to the boundary and
     its true size is unconstrained. Those frames (and the -1 "no sizing"
     sentinel, which sits below the grid) are excluded.

  2. A 2D histogram of scale_xy vs scale_z, as the DAMNIT per-run
     make_sizing_histogram does, but integrated over the chosen runs.

The scale data is read from the per-run CXI files written by the sizing job:

    entry_1/result_2/frames        indices of sized frames
    entry_1/result_2/data          per-frame scale [x, y, z]
    entry_1/result_2/scale_unique  the (x, y, z) search grid

Usage:
    python show_scale_histogram.py                       # all r####_hits.cxi
    python show_scale_histogram.py 12 13 14              # only these runs
    python show_scale_histogram.py 12 13 --title "Ery 0.75 nm"
"""

import os
import re
import glob
import argparse

import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

PREFIX = os.environ["EXP_PREFIX"]

out_1d = f'{PREFIX}/scratch/log/scale_histogram.pdf'
out_2d = f'{PREFIX}/scratch/log/scale_histogram_2d.pdf'

RESULT = 'entry_1/result_2'

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('runs', type=int, nargs='*',
                    help='run numbers to include (default: all)')
parser.add_argument('--title', default=None,
                    help='override the plot title (applied to both figures)')
args = parser.parse_args()

wanted = set(args.runs)

fnams = sorted(glob.glob(f'{PREFIX}/scratch/saved_hits/r[0-9][0-9][0-9][0-9]_hits.cxi'))

averages = []   # (scale_xy + scale_z)/2 for every non-boundary frame, all runs
all_xy   = []   # scale_xy for every sized frame, all runs (for the 2D plot)
all_z    = []   # scale_z  for every sized frame, all runs (for the 2D plot)
n_total = 0
n_boundary = 0
avg_step = None      # spacing of the discrete average values (grid step / 2)
xy_vals = z_vals = None   # the search grid (assumed shared across runs)

for fnam in tqdm(fnams):
    run = int(re.search(r'r(\d{4})', os.path.basename(fnam)).group(1))
    if wanted and run not in wanted:
        continue

    try:
        with h5py.File(fnam) as f:
            if f'{RESULT}/frames' not in f:
                continue
            frames       = f[f'{RESULT}/frames'][()]
            scale_d      = f[f'{RESULT}/data'][()][frames]      # (N, 3) -> [x, y, z]
            scale_unique = f[f'{RESULT}/scale_unique'][()]      # grid combos
    except (OSError, KeyError) as e:
        # truncated / corrupted / still-being-written CXI file: skip it
        tqdm.write(f'skipping {fnam}: {e}')
        continue

    if len(frames) == 0:
        continue

    scale_xy = scale_d[:, 0]
    scale_z  = scale_d[:, 2]

    # search grid (lowest / highest value per axis); assumed the same for all runs
    grid = scale_unique.T
    xy_vals = np.unique(grid[0])
    z_vals  = np.unique(grid[2])
    xy_lo, xy_hi = xy_vals[0], xy_vals[-1]
    z_lo,  z_hi  = z_vals[0],  z_vals[-1]

    # averaging two values on a step-d grid puts the average on a step-d/2 grid
    if avg_step is None:
        avg_step = min(np.diff(xy_vals).min(), np.diff(z_vals).min()) / 2

    # Drop frames at or beyond the grid edge in either axis. Using <= / >=
    # (rather than isclose to the edge) also throws out the "no sizing"
    # sentinel (-1), which sits below the low grid value.
    tol = avg_step / 10
    on_boundary = (
        (scale_xy <= xy_lo + tol) | (scale_xy >= xy_hi - tol) |
        (scale_z  <= z_lo  + tol) | (scale_z  >= z_hi  - tol)
    )

    keep = ~on_boundary
    n_total    += len(frames)
    n_boundary += int(on_boundary.sum())

    # both plots use the same boundary-excluded frames
    all_xy.append(scale_xy[keep])
    all_z.append(scale_z[keep])
    averages.append((scale_xy[keep] + scale_z[keep]) / 2)

averages = np.concatenate(averages) if averages else np.array([])
all_xy   = np.concatenate(all_xy)   if all_xy   else np.array([])
all_z    = np.concatenate(all_z)    if all_z    else np.array([])

print(f'{n_total} sized frames, {n_boundary} on a scale boundary (excluded), '
      f'{len(averages)} kept')

if len(averages) == 0:
    raise SystemExit('no sized frames found for the requested runs')

run_suffix = f'\nruns {sorted(wanted)}' if wanted else ''


# ---- 1D: average scale factor ------------------------------------------------
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
fig.set_tight_layout(True)

# bins aligned to the discrete average values (one bin per possible value) so
# there are no empty gaps between occupied bins
bins = np.arange(averages.min() - avg_step / 2,
                 averages.max() + avg_step,
                 avg_step)

ax.hist(averages, bins=bins, density=True,
        color='lightcoral', edgecolor='k', linewidth=0.5)
ax.set_xlabel('average scale factor  (scale_xy + scale_z) / 2')
ax.set_ylabel('probability density')
ax.set_title(args.title if args.title is not None
             else 'average sizing scale factor (boundary-pinned frames excluded)'
                  + run_suffix)
ax.spines[['right', 'top']].set_visible(False)

plt.savefig(out_1d)
print(f'wrote {out_1d}')


# ---- 2D: scale_xy vs scale_z (DAMNIT-style, integrated over runs) ------------
dx = xy_vals[1] - xy_vals[0]
dz = z_vals[1]  - z_vals[0]
bins_xy = dx * np.arange(len(xy_vals) + 1) + xy_vals[0] - dx / 2
bins_z  = dz * np.arange(len(z_vals)  + 1) + z_vals[0]  - dz / 2

hist, _, _ = np.histogram2d(all_xy, all_z, (bins_xy, bins_z))

fig2, ax2 = plt.subplots(1, 1)
fig2.set_size_inches(8, 8)
fig2.set_tight_layout(True)

im = ax2.imshow(hist, extent=[bins_z[0], bins_z[-1], bins_xy[0], bins_xy[-1]],
                origin='lower', aspect='auto')
fig2.colorbar(im, ax=ax2, label='number of frames')
ax2.set_xlabel('scale z')
ax2.set_ylabel('scale x,y')
ax2.set_title(args.title if args.title is not None
              else 'sizing scale histogram (integrated over runs)' + run_suffix)
ax2.spines[['right', 'top']].set_visible(False)

plt.savefig(out_2d)
print(f'wrote {out_2d}')

plt.show()
