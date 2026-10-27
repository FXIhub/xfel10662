"""
Per-event beam-fluence summaries combined over a set of Ery runs.

Reads the per-event incident fluence written back into each hit CXI by
offline/run_emc_fluence.py:

    entry_1/instrument_1/source_1/fluence_sizing   incident fluence [photons/m^2]
    entry_1/result_2/w_d                           refined per-frame fluence weight
    entry_1/instrument_1/source_1/pulse_energy     XGM pulse energy [J] (cross-check)

Both per-event arrays are aligned to the CXI events with -1 marking events that
were not sized; those are dropped here.

Three things are produced under {EXP_PREFIX}/scratch/log/:
  1. fluence_table.csv / fluence_table.h5 -- one row per sized event
     (run, event, w_d, fluence_m2, fluence_um2, pulse_energy_J).
  2. fluence_histogram.pdf -- histogram of the per-event fluence (photons/um^2,
     log-spaced bins) over all selected runs.
  3. fluence_vs_run.pdf -- per-run median fluence (16-84% band) vs run number,
     with the per-run median XGM pulse energy overlaid (a slope/correlation here
     reflects the effective focal area).

Usage
-----
    python show_fluence_histogram.py                       # all r####_hits.cxi
    python show_fluence_histogram.py 12 13 14              # only these runs
    python show_fluence_histogram.py --sample Erythrocruorin
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

out_table_csv = f'{PREFIX}/scratch/log/fluence_table.csv'
out_table_h5  = f'{PREFIX}/scratch/log/fluence_table.h5'
out_hist      = f'{PREFIX}/scratch/log/fluence_histogram.pdf'
out_vs_run    = f'{PREFIX}/scratch/log/fluence_vs_run.pdf'

FLUENCE = 'entry_1/instrument_1/source_1/fluence_sizing'
W_D     = 'entry_1/result_2/w_d'
PULSE   = 'entry_1/instrument_1/source_1/pulse_energy'
SAMPLE  = 'entry_1/sample_1/name'


def sample_of(f):
    """Sample name from an open CXI (sample_1/name), or None if absent."""
    if SAMPLE not in f:
        return None
    v = f[SAMPLE][()]
    return v.decode() if isinstance(v, bytes) else str(v)


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('runs', type=int, nargs='*',
                    help='run numbers to include (default: all)')
parser.add_argument('--title', default=None,
                    help='override the plot title (applied to both figures)')
parser.add_argument('--sample', action='append', default=[],
                    help='only include runs whose CXI sample_1/name matches '
                         '(repeatable)')
args = parser.parse_args()

wanted = set(args.runs)
want_samples = set(args.sample)

fnams = sorted(glob.glob(f'{PREFIX}/scratch/saved_hits/r[0-9][0-9][0-9][0-9]_hits.cxi'))

# per-event columns, accumulated across runs
col_run, col_event = [], []
col_wd, col_flu_m2, col_pulse = [], [], []
samples_seen = set()

for fnam in tqdm(fnams):
    run = int(re.search(r'r(\d{4})', os.path.basename(fnam)).group(1))
    if wanted and run not in wanted:
        continue

    try:
        with h5py.File(fnam) as f:
            if FLUENCE not in f:
                continue
            sample = sample_of(f)
            if want_samples and sample not in want_samples:
                continue
            flu = f[FLUENCE][()]                                   # (n_events,)
            wd  = f[W_D][()] if W_D in f else np.full_like(flu, -1.0)
            pe  = f[PULSE][()] if PULSE in f else np.full_like(flu, np.nan)
    except (OSError, KeyError) as e:
        tqdm.write(f'skipping {fnam}: {e}')
        continue

    sel = np.where(flu >= 0)[0]            # sized events with a valid fluence
    if sel.size == 0:
        continue

    samples_seen.add(sample)
    col_run.append(np.full(sel.size, run, dtype=np.int32))
    col_event.append(sel.astype(np.int64))
    col_wd.append(wd[sel].astype(np.float64))
    col_flu_m2.append(flu[sel].astype(np.float64))
    col_pulse.append(np.asarray(pe)[sel].astype(np.float64))

if not col_run:
    raise SystemExit('no sized events with a fluence_sizing dataset found '
                     '(run offline/run_emc_fluence.py first)')

run_a   = np.concatenate(col_run)
event_a = np.concatenate(col_event)
wd_a    = np.concatenate(col_wd)
flu_m2  = np.concatenate(col_flu_m2)
flu_um2 = flu_m2 * 1e-12
pulse_a = np.concatenate(col_pulse)

if not wanted and not want_samples and len(samples_seen) > 1:
    names = ', '.join(sorted(str(s) for s in samples_seen))
    print(f'WARNING: mixed {len(samples_seen)} samples ({names}); pass --sample '
          f'NAME or run numbers to separate them.')

print(f'{flu_m2.size} sized events across {len(np.unique(run_a))} runs; '
      f'median fluence {np.median(flu_um2):.3e} ph/um^2')


# ---- table: CSV + H5 ---------------------------------------------------------
header = 'run,event,w_d,fluence_m2,fluence_um2,pulse_energy_J'
table = np.column_stack([run_a, event_a, wd_a, flu_m2, flu_um2, pulse_a])
np.savetxt(out_table_csv, table, delimiter=',', header=header, comments='',
           fmt=['%d', '%d', '%.8e', '%.8e', '%.8e', '%.8e'])
print(f'wrote {out_table_csv}')

with h5py.File(out_table_h5, 'w') as f:
    f['run']            = run_a
    f['event']          = event_a
    f['w_d']            = wd_a
    f['fluence_m2']     = flu_m2
    f['fluence_um2']    = flu_um2
    f['pulse_energy_J'] = pulse_a
print(f'wrote {out_table_h5}')


sel = '' if not (wanted or want_samples) else (
    '\n' + '; '.join(filter(None, [
        f'runs {sorted(wanted)}' if wanted else '',
        ', '.join(sorted(want_samples)) if want_samples else ''])))


# ---- histogram of per-event fluence (log-spaced) -----------------------------
pos = flu_um2[flu_um2 > 0]
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
fig.set_tight_layout(True)
if pos.size:
    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 60)
    ax.hist(pos, bins=bins, color='steelblue', edgecolor='k', linewidth=0.3)
    ax.set_xscale('log')
ax.set_xlabel('incident fluence (photons / um^2)')
ax.set_ylabel('number of events')
ax.set_title(args.title if args.title is not None
             else 'per-event sizing fluence' + sel)
ax.spines[['right', 'top']].set_visible(False)
plt.savefig(out_hist)
print(f'wrote {out_hist}')


# ---- per-run trend, with XGM pulse energy overlaid ---------------------------
runs_u = np.unique(run_a)
med, lo, hi, pe_med = [], [], [], []
for r in runs_u:
    m = run_a == r
    p16, p50, p84 = np.percentile(flu_um2[m], [16, 50, 84])
    med.append(p50); lo.append(p50 - p16); hi.append(p84 - p50)
    pe = pulse_a[m]
    pe = pe[np.isfinite(pe)]
    pe_med.append(np.median(pe) if pe.size else np.nan)
med, lo, hi, pe_med = map(np.asarray, (med, lo, hi, pe_med))

fig2, ax2 = plt.subplots(1, 1)
fig2.set_size_inches(9, 6)
fig2.set_tight_layout(True)
ax2.errorbar(runs_u, med, yerr=[lo, hi], fmt='o-', color='steelblue',
             capsize=3, label='median fluence (16-84%)')
ax2.set_xlabel('run')
ax2.set_ylabel('median fluence (photons / um^2)', color='steelblue')
ax2.tick_params(axis='y', labelcolor='steelblue')
ax2.set_title(args.title if args.title is not None
              else 'per-run fluence vs XGM pulse energy' + sel)
ax2.spines[['top']].set_visible(False)

if np.any(np.isfinite(pe_med)):
    ax3 = ax2.twinx()
    ax3.plot(runs_u, pe_med * 1e6, 's--', color='darkorange', alpha=0.7,
             label='median XGM pulse energy')
    ax3.set_ylabel('XGM pulse energy (uJ)', color='darkorange')
    ax3.tick_params(axis='y', labelcolor='darkorange')
    ax3.spines[['top']].set_visible(False)

plt.savefig(out_vs_run)
print(f'wrote {out_vs_run}')

plt.show()
