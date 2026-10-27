"""
Fix pulse_energy and background_weighting in existing r{run}_hits.cxi files.

make_cxi_file.py used to index XGM data.intensityTD by the AGIPD pulseId, but
its columns are XGM pulse slots (mapping: LITFRM data.xgmPulseId). So most
frames got the energy of the wrong pulse, an empty slot (~μJ noise) or NaN
(AGIPD pulseId >= 1000 XGM columns). See make_cxi_file.load_pulse_energy.

A first version of this fix indexed xgmPulseId by frame slot instead of pulse
rank, which gives every frame the energy of the next pulse. Files in that state
(recognised by FRAME_SLOT_DESCRIPTION) are corrected the same way, with that
mapping as the previous pulse energy.

background_weighting = E_norm * a_t (add_background_cxi.compute_weighting)
inherited the bad E_norm. The per-train factor a_t is recovered exactly from the
sidecar as b_old / E_norm_old, so no VDS rescan is needed:
    b_new = E_norm_new * b_old / E_norm_old

Checks, per run, before writing (the run is skipped if either fails):
    - the reconstructed previous pulse energy matches the stored CXI pulse_energy
    - the recovered a_t is constant within each train

Updated in place:
    r{run}_hits.cxi         source_1/pulse_energy, detector_1/background_weighting
    r{run}_hits.cxi.bg.h5   background_weighting_per_vds_frame
Runs whose pulse_energy already carries the new description are skipped.
Merged files (merge_cxi.py output) are not touched; regenerate them.

    python fix_pulse_energy_cxi.py --dry-run 13 216     # report only
    python fix_pulse_energy_cxi.py -n 8                 # all runs in saved_hits
"""

import argparse
import glob
import multiprocessing as mp
import os
import re
import stat
import traceback
from contextlib import contextmanager

import numpy as np
import h5py
import extra_data

from constants import PREFIX
import make_cxi_file as mcf
import add_background_cxi as abg

PE_PATH = 'entry_1/instrument_1/source_1/pulse_energy'
BW_PATH = 'entry_1/instrument_1/detector_1/background_weighting'
VI_PATH = 'entry_1/instrument_1/detector_1/vds_index'
SIDECAR_BW = 'background_weighting_per_vds_frame'


# pulse_energy description written by the first, frame-slot version of the fix
FRAME_SLOT_DESCRIPTION = ('XGM pulse energy of the pulse that produced this frame '
                          '(XGM slot from LITFRM data.xgmPulseId); NaN if unavailable')


def frame_slot_pulse_energy(dc, a_train, a_pulse):
    """The first version of the fix: xgmPulseId indexed by frame slot."""
    lf = dc[mcf.LITFRM_SRC]
    lf_train, lf_pulse = mcf._pulse_resolved(lf['data.detectorPulseId'])
    _, lf_npulse = mcf._pulse_resolved(lf['data.nPulsePerFrame'])
    n_slots = lf['data.detectorPulseId'].ndarray().shape[1]
    lf_xgm_pulse = lf['data.xgmPulseId'].ndarray()[:, :n_slots].ravel()
    m, idx = mcf._align_by_pair(a_train, a_pulse, lf_train, lf_pulse)
    xgm_pulse = lf_xgm_pulse[idx].astype(np.int64)
    m &= lf_npulse[idx] > 0
    kd = dc[mcf.XGM_SRC, 'data.intensityTD']
    xgm_data = kd.ndarray()
    xgm_tid = np.asarray(kd.train_id_coordinates(), dtype=np.int64)
    order = np.argsort(xgm_tid)
    row = np.clip(np.searchsorted(xgm_tid[order], a_train), 0, len(xgm_tid) - 1)
    m &= xgm_tid[order][row] == a_train
    m &= (xgm_pulse >= 0) & (xgm_pulse < xgm_data.shape[1])
    E = np.full(len(a_train), np.nan, dtype=np.float32)
    E[m] = xgm_data[order[row[m]], xgm_pulse[m]] * 1e-6
    return E


def old_pulse_energy(dc, a_train, a_pulse):
    """The buggy make_cxi_file lookup: intensityTD[train, AGIPD pulseId]."""
    E = np.full(len(a_train), np.nan, dtype=np.float32)
    if mcf.XGM_SRC not in dc.all_sources:
        return E
    kd = dc[mcf.XGM_SRC, 'data.intensityTD']
    xgm_data = kd.ndarray()
    train_to_row = {int(t): i for i, t in enumerate(kd.train_id_coordinates())}
    for k in range(len(a_train)):
        row = train_to_row.get(int(a_train[k]))
        p = int(a_pulse[k])
        if row is not None and 0 <= p < xgm_data.shape[1]:
            E[k] = xgm_data[row, p] * 1e-6
    return E


@contextmanager
def writable(path):
    """Temporarily add u+w (the saved_hits files are read-only), then restore."""
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IWUSR)
    try:
        yield
    finally:
        os.chmod(path, stat.S_IMODE(mode))


def fix_run(run, proposal, dry_run):
    cxi_file     = f'{PREFIX}scratch/saved_hits/r{run:04d}_hits.cxi'
    vds_file     = f'{PREFIX}scratch/vds/r{run:04d}.cxi'
    sidecar_file = f'{cxi_file}.bg.h5'

    with h5py.File(cxi_file, 'r') as f:
        description = f[PE_PATH].attrs.get('description')
        if description == mcf.PULSE_ENERGY_DESCRIPTION:
            return f'r{run:04d} already fixed, skipped'
        vds_idx = f[VI_PATH][()].astype(np.int64)
        pe_stored = f[PE_PATH][()]
        has_bw = BW_PATH in f

    with h5py.File(vds_file, 'r') as g:
        a_train = g['/entry_1/trainId'][:].astype(np.int64)
        a_pulse = g['/entry_1/pulseId'][:].astype(np.int64)

    dc = extra_data.open_run(proposal, run, data='all')
    previous = (frame_slot_pulse_energy if description == FRAME_SLOT_DESCRIPTION
                else old_pulse_energy)
    E_old = previous(dc, a_train, a_pulse)
    E_new = mcf.load_pulse_energy(dc, a_train, a_pulse)

    # allow last-bit float32 differences: the files were written under a numpy
    # with different scalar promotion for float32 * 1e-6
    if not np.allclose(E_old[vds_idx], pe_stored, rtol=1e-6, atol=0, equal_nan=True):
        return f'r{run:04d} FAILED: reconstructed old pulse_energy != stored, skipped'

    b_new = None
    if has_bw and os.path.exists(sidecar_file):
        with h5py.File(sidecar_file, 'r') as s:
            b_old = s[SIDECAR_BW][()].astype(np.float64)
        a = b_old / abg.energy_norm(E_old)

        # a_t must be constant within a train (float32 rounding aside)
        order = np.argsort(a_train, kind='stable')
        t, a_s = a_train[order], a[order]
        starts = np.r_[0, np.nonzero(np.diff(t))[0] + 1]
        spread = np.maximum.reduceat(a_s, starts) - np.minimum.reduceat(a_s, starts)
        rel = np.max(spread / np.maximum(np.abs(a_s[starts]), 1e-12))
        if rel > 1e-5:
            return f'r{run:04d} FAILED: a_t varies within a train (rel {rel:.2g}), skipped'

        b_new = (abg.energy_norm(E_new) * a).astype(np.float32)
        b_msg = (f'bw mean {b_old[vds_idx].mean():.3f} -> {b_new[vds_idx].mean():.3f} '
                 f'std {b_old[vds_idx].std():.3f} -> {b_new[vds_idx].std():.3f}')
    else:
        b_msg = 'no background_weighting/sidecar, bw untouched'

    pc = lambda E: np.round(np.nanpercentile(E[vds_idx], [5, 50, 95]) * 1e6).astype(int) \
        if np.isfinite(E[vds_idx]).any() else 'all NaN'
    msg = (f'r{run:04d} pulse_energy uJ 5/50/95% {pc(E_old)} -> {pc(E_new)}, '
           f'NaN {np.isnan(E_old[vds_idx]).mean():.2f} -> {np.isnan(E_new[vds_idx]).mean():.2f}; {b_msg}')
    if dry_run:
        return msg + ' [dry run]'

    if b_new is not None:
        with writable(sidecar_file), h5py.File(sidecar_file, 'r+') as s:
            s[SIDECAR_BW][...] = b_new
    with writable(cxi_file), h5py.File(cxi_file, 'r+') as f:
        if b_new is not None:
            f[BW_PATH][...] = b_new[vds_idx]
        pe = f[PE_PATH]
        pe[...] = E_new[vds_idx]
        pe.attrs['source'] = mcf.XGM_SRC
        pe.attrs['key'] = 'data.intensityTD'
        pe.attrs['description'] = mcf.PULSE_ENERGY_DESCRIPTION
    return msg


def _worker(task):
    run, proposal, dry_run = task
    try:
        return fix_run(run, proposal, dry_run)
    except Exception:
        return f'r{run:04d} ERROR:\n{traceback.format_exc()}'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=int, nargs='*',
                   help='run numbers (default: every r*_hits.cxi in scratch/saved_hits)')
    p.add_argument('-n', '--nproc', type=int, default=1, help='runs processed in parallel')
    p.add_argument('--dry-run', action='store_true', help='report without writing')
    args = p.parse_args()

    runs = args.run or sorted(
        int(re.search(r'r(\d{4})_hits\.cxi$', fn).group(1))
        for fn in glob.glob(f'{PREFIX}scratch/saved_hits/r[0-9][0-9][0-9][0-9]_hits.cxi'))
    proposal = mcf.get_proposal_from_prefix()
    tasks = [(r, proposal, args.dry_run) for r in runs]

    with mp.Pool(args.nproc) as pool:
        for msg in pool.imap(_worker, tasks):
            print(msg, flush=True)


if __name__ == '__main__':
    main()
