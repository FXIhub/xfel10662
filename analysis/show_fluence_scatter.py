"""Scatter the per-event sizing fluence vs run number, coloured by the amount of
antifreeze added to the Ery sample (read from the DAMNIT run comments).

Companion to show_hit_score_scatter.py, but for the incident beam fluence written
back into the CXIs by offline/run_emc_fluence.py:

  fluence    source_1/fluence_sizing[fluence_sizing >= 0]   incident fluence
                                                            [photons/m^2], -1 ==
                                                            event not sized

Only good single-particle hits are shown: events with score/is_hit set (the
size/class/score filter from add_is_hit_cxi.py), which drops clusters/aggregates
that scatter more strongly and would inflate the per-event fluence estimate.

Each good-hit event becomes one point (fluence in photons/um^2 vs its run), coloured
by the antifreeze concentration parsed from that run's DAMNIT comment (e.g. "0.8 uM
Ery + 0.02% antifreeze ..." -> 0.02 %; "... 0 antifreeze" / "No antifreeze" -> 0 %;
runs whose comment does not state it -> "unknown"). The three Ery runs from the
previous beamtime p010996 (before the cryo shutdown) are shown in a separate band
on the left, to compare the peak fluence before vs after the shutdown. Click a
point (interactive mode) to display the corresponding frame.

Usage:
    python show_fluence_scatter.py                 # interactive, raw fluence
    python show_fluence_scatter.py background       # non-interactive, just the pdf
    python show_fluence_scatter.py background norm   # normalise by pulse energy
                                                     # (fluence / pulse energy, to
                                                     #  gauge the focal spot size)
"""

import os
import re
import sys
import glob
import sqlite3

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

args = sys.argv[1:]
run_non_interactive = any('background' in a for a in args)
normalise = any(a.startswith('norm') for a in args)

PREFIX = os.environ["EXP_PREFIX"]

out = (f'{PREFIX}/scratch/log/fluence_scatter_pernorm.pdf' if normalise
       else f'{PREFIX}/scratch/log/fluence_scatter.pdf')

SOURCE  = '/entry_1/instrument_1/source_1'
FLUENCE = f'{SOURCE}/fluence_sizing'
PULSE   = f'{SOURCE}/pulse_energy'                 # J, per event (XGM)
IS_HIT  = '/entry_1/instrument_1/detector_1/score/is_hit'

# DAMNIT database holding the per-run comments (override with $DAMNIT_DB)
DAMNIT_DB = os.environ.get(
    'DAMNIT_DB',
    '/gpfs/exfel/u/usr/SPB/202601/p010662/Shared/amore/runs.sqlite')

# Ery runs from the previous beamtime (before the cryo shutdown), shown in a band
# on the left. (cxi path, x position in the band, real run number)
BEFORE_LABEL = 'before cryo (p010996)'
BEFORE_P = '/gpfs/exfel/exp/SPB/202604/p010996/scratch/saved_hits'
BEFORE = [(f'{BEFORE_P}/r{r:04d}_hits.cxi', x, r)
          for x, r in zip((1, 2, 3), (26, 27, 28))]
DIVIDER_X = 7          # vertical divider between the before-cryo band and p10662


def antifreeze_label(comment):
    """Legend label for the antifreeze stated in a DAMNIT run comment, or None
    when the comment doesn't mention it.

    Returns a concentration label "<x>% antifreeze" when a percentage is given
    ("0.02% antifreeze" / "0.4% anti-freeze" / typo "antifeeze"), "0% antifreeze"
    for the explicit "0 antifreeze" / "No antifreeze", and descriptive labels for
    the runs with an uncertain amount: "added antifreeze (amount?)" and
    "<n>x diluted" (per the beamtime notes a 5x dilution means ~5x less
    antifreeze)."""
    if not comment:
        return None
    c = comment.lower()
    if 'anti' in c or 'freez' in c:
        m = re.search(r'([0-9]*\.?[0-9]+)\s*%', c)          # "0.02%"
        if m:
            return f'{float(m.group(1)):g}% antifreeze'
        if re.search(r'(\b0\s+anti|\b0\s+freez|no\s+ani|no\s+anti|no\s+freez)', c):
            return '0% antifreeze'
        if 'added' in c:
            return 'added antifreeze (amount?)'
        return 'antifreeze (amount?)'
    # dilution comments (no explicit antifreeze, but treated as reduced antifreeze)
    m = re.search(r'dilut\w*\s*([0-9]+)\s*x|([0-9]+)\s*x\s*dilut', c)
    if m:
        return f'{m.group(1) or m.group(2)}x diluted'
    if 'dilut' in c:
        return 'diluted'
    return None


def run_antifreeze(db=DAMNIT_DB):
    """run -> antifreeze label from the DAMNIT per-run comments (name='comment' in
    run_variables); empty dict if the database can't be read."""
    try:
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        rows = con.execute(
            "SELECT run, value FROM run_variables WHERE name='comment'").fetchall()
        con.close()
    except sqlite3.Error as e:
        print(f'warning: could not read DAMNIT comments from {db}: {e}')
        return {}
    out = {}
    for run, comment in rows:
        lbl = antifreeze_label(comment)
        if lbl is not None:
            out[run] = lbl
    return out


af_map = run_antifreeze()


def af_label(run):
    return af_map.get(run, 'unknown')


def af_value(lbl):
    """The numeric antifreeze % of a label, or None for the descriptive labels
    ('added antifreeze (amount?)', '5x diluted', 'before cryo ...', 'unknown')."""
    try:
        return float(lbl.split('%')[0])
    except ValueError:
        return None


# The per-event pulse energy is bimodal within a run: the real beam (a few mJ) plus
# a dropout population pinned near the XGM noise floor (~1 uJ). The median can fall
# in the gap, so the clamp references a HIGH percentile (the real beam): readings
# below PULSE_CLAMP_FRAC x that are treated as dropouts (interpolated, then floored),
# which collapses the spurious high-ratio band in fluence/pulse_energy.
PULSE_REF_PCT = 90
PULSE_CLAMP_FRAC = 0.5


def fill_pulse_energy(pe, ref_pct=PULSE_REF_PCT, clamp_frac=PULSE_CLAMP_FRAC):
    """Per-event pulse energy [J] with missing/zero entries and the sub-floor XGM
    dropouts filled by linear interpolation over event order and clamped to a floor
    of clamp_frac x the run's ref_pct-percentile (the real-beam level). All NaN if
    no event has a usable pulse energy."""
    pe = np.asarray(pe, dtype=float)
    pos = np.isfinite(pe) & (pe > 0)
    if pos.sum() == 0:
        return np.full(pe.shape, np.nan)
    floor = clamp_frac * np.percentile(pe[pos], ref_pct)
    valid = pos & (pe >= floor)                 # drop the noise-floor dropouts
    if valid.sum() == 0:
        valid = pos
    if valid.sum() == 1:
        out = np.where(valid, pe, pe[valid][0])
    else:
        out = pe.copy()
        idx = np.arange(pe.size)
        out[~valid] = np.interp(idx[~valid], idx[valid], pe[valid])
    return np.clip(out, floor, None)


# accumulators keyed by group label
xs      = {}   # x position per point (run number, or band position for p010996)
flu     = {}   # fluence (photons/um^2) per point
pe      = {}   # pulse energy (J) per point, interpolated over gaps
files   = {}   # cxi file per point (for interactive frame display)
indexes = {}   # frame index within the cxi per point
realrun = {}   # true run number per point (for the click message)

# sources: (cxi path, x position, real run, forced label or None -> antifreeze)
sources = []
for fn in sorted(glob.glob(f'{PREFIX}/scratch/saved_hits/r[0-9][0-9][0-9][0-9]_hits.cxi')):
    run = int(re.search(r'r(\d{4})', os.path.basename(fn)).group(1))
    sources.append((fn, run, run, None))
sources += [(fn, x, r, BEFORE_LABEL) for fn, x, r in BEFORE]

for fnam, x, run, forced in tqdm(sources):
    try:
        with h5py.File(fnam) as f:
            if FLUENCE not in f:
                continue
            if IS_HIT not in f:
                tqdm.write(f'skipping {fnam}: no {IS_HIT} (run good_hits first)')
                continue
            fluence = f[FLUENCE][()]
            is_hit  = f[IS_HIT][()].astype(bool)
            pulse   = (fill_pulse_energy(f[PULSE][()]) if PULSE in f
                       else np.full(fluence.shape, np.nan))
            # good single-particle hits only: is_hit drops clusters/aggregates that
            # scatter more strongly and would inflate the per-event fluence.
            ds = np.where((fluence >= 0) & is_hit)[0]
            if len(ds) == 0:
                continue
    except (OSError, KeyError) as e:
        # truncated / corrupted / still-being-written CXI file: skip it
        tqdm.write(f'skipping {fnam}: {e}')
        continue

    name = forced if forced is not None else af_label(run)
    for d in (xs, flu, pe, files, indexes, realrun):
        d.setdefault(name, [])

    f_um2 = fluence[ds] * 1e-12                      # photons/m^2 -> photons/um^2
    for d, fv, pv in zip(ds, f_um2, pulse[ds]):
        xs[name].append(x)
        flu[name].append(fv)
        pe[name].append(pv)
        files[name].append(fnam)
        indexes[name].append(int(d))
        realrun[name].append(run)


def yvalue(name):
    """Plotted y: raw fluence [ph/um^2], or fluence / pulse energy [ph/um^2/mJ]
    when normalising (pulse energy J -> mJ); NaN where pulse energy is missing."""
    f_um2 = np.asarray(flu[name], dtype=float)
    if not normalise:
        return f_um2
    p_mJ = np.asarray(pe[name], dtype=float) * 1e3
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(p_mJ > 0, f_um2 / p_mJ, np.nan)


def load_frame(fnam, index):
    with h5py.File(fnam) as f:
        return f['entry_1/data_1/data'][index]


if not run_non_interactive and xs:
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
        run   = realrun[name][i]
        fnam  = files[name][i]
        index = indexes[name][i]
        print(f'clicked on run {run}, fnam {fnam}, index {index}')
        frame_plot.set_data(geom.position_modules(load_frame(fnam, index))[0] ** 0.2)
        fig_im.canvas.draw()
        fig_im.canvas.flush_events()


# plot
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(12, 8)
fig.set_tight_layout(True)

# order the legend: before-cryo band, then increasing antifreeze %, then the
# uncertain-amount labels, then unknown; let matplotlib pick the colours.
numeric_labels = sorted((l for l in xs if af_value(l) is not None), key=af_value)
special_labels = sorted(l for l in xs
                        if af_value(l) is None and l not in ('unknown', BEFORE_LABEL))
order = ([BEFORE_LABEL] if BEFORE_LABEL in xs else []) + numeric_labels \
        + special_labels + (['unknown'] if 'unknown' in xs else [])

for name in order:
    art = ax.scatter(xs[name], yvalue(name), alpha=0.8, s=5.0, picker=5, label=name)
    art.name = name

# per-run PEAK (99th percentile) marker. The median/bulk just tracks the hit-score
# threshold (it sets where the weak-hit tail is cut), so it is not meaningful; the
# upper envelope is threshold-independent and reflects the focal core, which is the
# quantity to compare before vs after the shutdown.
all_x, all_y = [], []
for name in order:
    all_x.append(np.asarray(xs[name], dtype=float))
    all_y.append(yvalue(name))
all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
ok = np.isfinite(all_y)
all_x, all_y = all_x[ok], all_y[ok]
ux = np.unique(all_x)
peak = np.array([np.percentile(all_y[all_x == x], 99) for x in ux])
ax.scatter(ux, peak, marker='o', s=6, color='k', zorder=5,
           label='per-run 99th-pct (peak)')

# divider + labels for the before/after-cryo split
if BEFORE_LABEL in xs:
    ax.axvline(DIVIDER_X, color='0.5', lw=1, ls='--')

ax.set_yscale('log')
ylim = ax.get_ylim()
if normalise:
    ax.set_ylabel('fluence / pulse energy (photons / um^2 / mJ)')
else:
    ax.set_ylim([1e11, ylim[1]])
    ax.set_ylabel('incident fluence (photons / um^2)')

if BEFORE_LABEL in xs:
    ytop = ax.get_ylim()[1]
    ax.text(2, ytop, 'before cryo\n(p010996 r26-28)', ha='center', va='top',
            fontsize=9, color='0.3')

ax.legend(markerscale=5, loc='upper right', title='Ery antifreeze')
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('run number  (p010996 r26-28 shown in the left band)')
ax.set_title(('per-event fluence/pulse-energy of good hits (is_hit) vs run'
              if normalise else
              'per-event fluence of good hits (is_hit) vs run')
             + ', coloured by antifreeze added to Ery (DAMNIT comments)', fontsize=12)
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.set_axisbelow(True)          # draw the grid behind the scatter points
ax.grid(visible=True, which='both', alpha=1.0)

if not run_non_interactive and xs:
    fig.canvas.callbacks.connect('pick_event', on_pick)

fig.show()
plt.savefig(out)
print(f'wrote {out}')

if not run_non_interactive:
    plt.show()
