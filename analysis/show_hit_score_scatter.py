"""Scatter the per-shot hit_score_mask of good hits vs run number, coloured by
sample, with the beamline-background level overlaid per run.

This is the current-beamtime replacement for show_peaks_interactive.py. Instead of
re-summing photons from the (size-filtered) frames, it reads the precomputed
per-shot score from the CXI files:

  good hits      score/hit_score_mask[score/is_hit]   (is_hit already encodes the
                                                       size + class + sigma filter
                                                       written by add_is_hit_cxi.py)
  background     score/hit_score_mask_data_white       sum_pix(hit_mask * data_white)

Click a point (interactive mode) to display the corresponding frame.

Usage:
    python show_hit_score_scatter.py            # interactive
    python show_hit_score_scatter.py background # non-interactive, just write the pdf
"""

import os
import re
import sys
import glob

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

run_non_interactive = len(sys.argv) > 1 and 'background' in sys.argv[1]

PREFIX = os.environ["EXP_PREFIX"]

out = f'{PREFIX}/scratch/log/hit_score_report.pdf'

DET   = '/entry_1/instrument_1/detector_1'
SCORE = f'{DET}/score'

# accumulators keyed by sample name
runs    = {}   # run number per point
scores  = {}   # hit_score_mask per point
files   = {}   # cxi file per point (for interactive frame display)
indexes = {}   # frame index within the cxi per point
back_line = {} # run -> background level (one value per run)

# only the per-run files r####_hits.cxi (skip merged *_all_hits.cxi and the like)
fnams = sorted(glob.glob(f'{PREFIX}/scratch/saved_hits/r[0-9][0-9][0-9][0-9]_hits.cxi'))

for fnam in tqdm(fnams):
    run = int(re.search(r'r(\d{4})', os.path.basename(fnam)).group(1))
    try:
        with h5py.File(fnam) as f:
            # background level for this run: raw sum_pix(hit_mask * data_white)
            if f'{SCORE}/hit_score_mask_data_white' in f:
                back_line[run] = float(f[f'{SCORE}/hit_score_mask_data_white'][()])

            if f'{SCORE}/is_hit' not in f or f'{SCORE}/hit_score_mask' not in f:
                continue

            is_hit = f[f'{SCORE}/is_hit'][()].astype(bool)
            ds = np.where(is_hit)[0]
            if len(ds) == 0:
                continue

            score = f[f'{SCORE}/hit_score_mask'][()]
            name  = f['entry_1/sample_1/name'][()].decode('utf-8')
    except (OSError, KeyError) as e:
        # truncated / corrupted / still-being-written CXI file: skip it
        tqdm.write(f'skipping {fnam}: {e}')
        continue

    runs.setdefault(name, [])
    scores.setdefault(name, [])
    files.setdefault(name, [])
    indexes.setdefault(name, [])

    for d in ds:
        runs[name].append(run)
        scores[name].append(score[d])
        files[name].append(fnam)
        indexes[name].append(int(d))


def load_frame(fnam, index):
    with h5py.File(fnam) as f:
        return f['entry_1/data_1/data'][index]


if not run_non_interactive and runs:
    import extra_geom
    geom_fnam = sorted(glob.glob('../geom/r*.geom'))[0]
    geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom_fnam)

    fig_im, ax_im = plt.subplots(figsize=(10, 10))
    fig_im.set_tight_layout(True)

    name0 = next(iter(files))
    frame_plot = ax_im.imshow(
        geom.position_modules(load_frame(files[name0][0], indexes[name0][0]))[0] ** 0.2,
        vmin=0)
    fig_im.show()

    def on_pick(event):
        i     = event.ind[0]
        name  = event.artist.name
        run   = runs[name][i]
        fnam  = files[name][i]
        index = indexes[name][i]
        print(f'clicked on run {run}, fnam {fnam}, index {index}')
        frame_plot.set_data(geom.position_modules(load_frame(fnam, index))[0] ** 0.2)
        fig_im.canvas.draw()
        fig_im.canvas.flush_events()


# plot
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(30, 8)
fig.set_tight_layout(True)

for name in runs.keys():
    art = ax.scatter(runs[name], scores[name], alpha=0.6, s=3.0, picker=5, label=name)
    art.name = name

if back_line:
    r = sorted(back_line)
    v = [back_line[i] for i in r]
    ax.scatter(r, v, alpha=0.6, c='k', s=3.0, label='background')

# keep the y-axis off zero for the log scale
ylim = ax.get_ylim()
ax.set_ylim([max(1.0, ylim[0]), ylim[1]])

ax.legend(markerscale=5, loc='upper left')
ax.set_yscale('log')
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('run number')
ax.set_ylabel('hit_score_mask (photons in masked region)')
ax.set_title('hit_score_mask of good hits (is_hit) vs run, coloured by sample\n'
             'black = beamline background level (hit_score_mask of data_white)', fontsize=12)
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.grid(visible=True, which='both', alpha=0.3)

if not run_non_interactive and runs:
    fig.canvas.callbacks.connect('pick_event', on_pick)

fig.show()
plt.savefig(out)

if not run_non_interactive:
    plt.show()
