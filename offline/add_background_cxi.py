"""
Compute a per-shot background weighting from non-hit frames and add it to the
CXI file produced by make_cxi_file.py.

The background is the beamline-only signal — gas scattering, fluorescence and
stray light — estimated as the mean of non-hit frames. CXI spec field
data_white ("Image recorded without the sample") is the closest match.

Per-shot metadata (pulse_energy, trainId, hit/miss, photons) come from extra_data
via the same facility sources as make_cxi_file.py.

The per-pixel background is computed in a single pass over non-hit frames in the
VDS, accumulating per-memory-cell sums; the cells are then aggregated per pixel
with median + MAD-based rejection so bad-pixel-in-some-cells are dropped before
averaging. `--max-trains N` limits the number of trains processed for fast
semi-realtime feedback during an experiment.

Runs in two phases so the slow VDS scan can happen in parallel with
make_cxi_file.py without two writers touching the CXI file:

  default (compute):  read VDS, write sidecar  r{run}_hits.cxi.bg.h5  with
                          data_white                          (NMODULES, ss, fs)
                          background_weighting_per_vds_frame  (Nvds,)
  --merge:            open the CXI in r+, copy data_white and
                      background_weighting = b[vds_index] into
                          /entry_1/instrument_1/detector_1/data_white
                          /entry_1/instrument_1/detector_1/background_weighting
"""

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import h5py
import extra_data
from tqdm import tqdm

from constants import PREFIX
import make_cxi_file as mcf


def _aggregate_module(cell_mean_m, k_sigma):
    """Per-pixel median + MAD-based survivor mean across cells for one module.
    cell_mean_m shape (Ncells, ss, fs)."""
    valid = ~np.isnan(cell_mean_m)
    med = np.nanmedian(cell_mean_m, axis=0)
    mad = np.nanmedian(np.abs(cell_mean_m - med), axis=0)
    sig = 1.4826 * mad
    outlier = np.abs(cell_mean_m - med) > (k_sigma * sig + np.finfo(np.float32).eps)
    survivors = valid & ~outlier
    n_surv = survivors.sum(axis=0)
    surv_sum = np.where(survivors, cell_mean_m, 0.0).sum(axis=0)
    return np.where(n_surv > 0, surv_sum / np.maximum(n_surv, 1), med).astype(np.float32)


def _accumulate_chunk(task):
    """Read assigned non-hit frames from the VDS and return per-cell sum + count
    for the worker's cell partition, plus the total counts of each frame read.
    No two workers touch the same cellId. The per-frame totals (paired with their
    global VDS index) feed the per-shot weighting, computed from the same VDS data
    as the background so the two are on a consistent scale."""
    rank, vds_file, vds_path, miss_idx, miss_cells, my_cell_ids, frame_shape = task
    cid_to_local = {int(c): i for i, c in enumerate(my_cell_ids)}
    sums  = np.zeros((len(my_cell_ids),) + frame_shape, dtype=np.float32)
    count = np.zeros(len(my_cell_ids), dtype=np.int64)
    frame_totals = np.empty(len(miss_idx), dtype=np.float64)
    it = (tqdm(range(len(miss_idx)), desc=f'rank {rank} reading non-hits')
          if rank == 0 else range(len(miss_idx)))
    with h5py.File(vds_file) as g:
        data = g[vds_path]
        for i in it:
            j = cid_to_local[int(miss_cells[i])]
            frame = np.squeeze(data[miss_idx[i]]).astype(np.float32)
            sums[j]  += frame
            count[j] += 1
            frame_totals[i] = np.nansum(frame)
    return my_cell_ids, sums, count, miss_idx, frame_totals


def compute_background_from_misses(vds_file, ev, max_trains=None, k_sigma=5.0,
                                   nproc=1):
    """Single-pass per-pixel background averaged over memory cells, with outlier
    rejection across cells per pixel.

    1. Pick non-hit frames in the first `max_trains` trains (or all trains).
    2. Stream each frame from the VDS; accumulate sum and count per memory cell.
       Parallelised by partitioning cellIds across `nproc` workers so frame IO is
       balanced and the workers' output buffers don't overlap.
    3. Per-cell mean = sum / count.
    4. Per pixel across cells: drop cells whose mean is >k_sigma * (1.4826 * MAD)
       from the median across cells, then average the survivors.

    Returns (background (NMODULES, ss, fs) float32, n_frames_used int,
             frame_counts (Nframes,) float64 — total counts of each miss frame,
             NaN for hits/unread frames).
    """
    train  = np.asarray(ev['trainId'])
    cell   = np.asarray(ev['cellId'])
    misses = ~np.asarray(ev['is_hit'], dtype=bool)

    if max_trains is not None:
        kept_trains = np.unique(train)[:max_trains]
        misses &= np.isin(train, kept_trains)

    miss_idx = np.where(misses)[0]
    if len(miss_idx) == 0:
        raise RuntimeError('no non-hit frames available for background computation')
    miss_cells = cell[miss_idx]
    n_cells = int(miss_cells.max()) + 1
    unique_cids = np.unique(miss_cells)
    print(f'background from {len(miss_idx)} non-hit frames across '
          f'{len(np.unique(train[miss_idx]))} trains, {len(unique_cids)} cells '
          f'(nproc={nproc})')

    with h5py.File(vds_file) as g:
        frame_shape = g[mcf.VDS_DATA_PATH].shape[1:]

    # Partition cellIds across workers; each worker reads only the frames belonging
    # to its cell set and writes to disjoint buffer rows.
    nproc = max(1, int(nproc))
    cell_chunks = np.array_split(unique_cids, nproc)
    tasks = []
    for rank, my_cids in enumerate(cell_chunks):
        mask = np.isin(miss_cells, my_cids)
        tasks.append((rank, vds_file, mcf.VDS_DATA_PATH,
                      miss_idx[mask], miss_cells[mask], np.asarray(my_cids),
                      frame_shape))

    if nproc == 1:
        results = [_accumulate_chunk(tasks[0])]
    else:
        with mp.Pool(nproc) as pool:
            results = pool.map(_accumulate_chunk, tasks)

    # Assemble the per-cell accumulator from the per-worker partitions, and the
    # per-VDS-frame total counts (the weighting numerator, see compute_weighting).
    sums  = np.zeros((n_cells,) + frame_shape, dtype=np.float32)
    count = np.zeros(n_cells, dtype=np.int64)
    frame_counts = np.full(train.shape[0], np.nan, dtype=np.float64)
    for cids, s, c, idx, totals in results:
        idx_global = np.asarray(cids, dtype=int)
        sums[idx_global]  = s
        count[idx_global] = c
        frame_counts[idx] = totals

    has_data = count > 0
    cell_mean = np.full(sums.shape, np.nan, dtype=np.float32)
    cell_mean[has_data] = sums[has_data] / count[has_data, None, None, None]

    # Robust per-pixel aggregation across cells, parallel across modules.
    # numpy's median releases the GIL, so threads are sufficient and avoid the
    # IPC cost of shipping cell_mean to subprocesses.
    n_modules = cell_mean.shape[1]
    if nproc == 1:
        per_module = [_aggregate_module(cell_mean[:, m], k_sigma)
                      for m in range(n_modules)]
    else:
        with ThreadPoolExecutor(max_workers=min(nproc, n_modules)) as ex:
            per_module = list(ex.map(
                lambda m: _aggregate_module(cell_mean[:, m], k_sigma),
                range(n_modules)))
    background = np.stack(per_module, axis=0)
    return background, int(len(miss_idx)), frame_counts


def compute_weighting(ev, frame_counts, back_counts, emin=1e-3):
    """Return the per-VDS-frame background weighting b_d.

    For each train t:
        a_t = (sum of miss-frame total counts in t) / (<B> * Nmiss_t)
    Per-shot:
        b_d = (E_d / <E>) * a_(t_d)

    `frame_counts` holds the total counts of each miss frame, summed from the
    same VDS data as the background image, so the numerator and denominator <B>
    (= back_counts = sum of data_white) are on the same scale and a_t averages
    to ~1. Using the separate LITFRM proxy here put them on different scales and
    drove the mean weighting far below 1.
    """
    E       = np.asarray(ev['pulse_energy'])
    tids    = np.asarray(ev['trainId'])
    misses  = ~np.asarray(ev['is_hit'], dtype=bool)
    Kphot   = np.asarray(frame_counts)

    a_d = -np.ones(tids.shape[0], dtype=float)
    for t in np.unique(tids):
        m = np.where(tids == t)[0]
        n = m[misses[m]]
        if len(n) > 0 and np.all(np.isfinite(Kphot[n])):
            a_d[m] = np.sum(Kphot[n]) / (back_counts * len(n))

    good_e = np.isfinite(E) & (E > emin)
    if good_e.any():
        E_norm = np.where(good_e, E / E[good_e].mean(), 1.0)
    else:
        E_norm = np.ones_like(E)
    return (E_norm * a_d).astype(np.float32)


def write_sidecar(sidecar_file, background, weighting_per_vds_frame):
    """Persist the slow-to-compute artefacts so a later --merge step can patch
    them into the CXI file without contending with make_cxi_file.py for the
    HDF5 write lock."""
    with h5py.File(sidecar_file, 'w') as f:
        ds = f.create_dataset('data_white', data=background,
                              compression='gzip', compression_opts=1, shuffle=True)
        ds.attrs['units'] = 'counts'
        f.create_dataset('background_weighting_per_vds_frame',
                         data=weighting_per_vds_frame,
                         compression='gzip', compression_opts=1, shuffle=True)


def merge_sidecar_into_cxi(sidecar_file, cxi_file):
    with h5py.File(sidecar_file, 'r') as s:
        back = s['data_white'][()]
        b    = s['background_weighting_per_vds_frame'][()]

    with h5py.File(cxi_file, 'r+') as f:
        det = f['entry_1/instrument_1/detector_1']

        if 'data_white' in det:
            del det['data_white']
        dw = det.create_dataset('data_white', data=back,
                                compression='gzip', compression_opts=1,
                                shuffle=True)
        dw.attrs['axes']  = 'module_identifier:y:x'
        dw.attrs['units'] = 'counts'

        vds_idx = det['vds_index'][()]
        if 'background_weighting' in det:
            del det['background_weighting']
        bw = det.create_dataset('background_weighting',
                                data=b[vds_idx],
                                compression='gzip', compression_opts=1,
                                shuffle=True)
        bw.attrs['axes'] = 'experiment_identifier'


def main():
    parser = argparse.ArgumentParser(
        description='Compute beamline background + per-shot weighting and either '
                    'write a sidecar (default) or merge an existing sidecar into '
                    'the CXI file (--merge).')
    parser.add_argument('run', type=int, nargs='+', help='Run number/s')
    parser.add_argument('--merge', action='store_true',
                        help='Skip the compute step and merge the existing '
                             'sidecar into the CXI file. Use after make_cxi_file.py '
                             'has finished writing.')
    parser.add_argument('--max-trains', type=int, default=None,
                        help='cap on trains processed for the background mean '
                             '(speeds up semi-realtime feedback; default: all)')
    parser.add_argument('--k-sigma', type=float, default=5.0,
                        help='MAD threshold for rejecting outlier cells per '
                             'pixel before averaging (default: 5)')
    parser.add_argument('-n', '--nproc', type=int, default=1,
                        help='number of processes reading non-hit frames in parallel')
    args = parser.parse_args()

    proposal = mcf.get_proposal_from_prefix()

    for run in tqdm(args.run):
        cxi_file     = f'{PREFIX}scratch/saved_hits/r{run:04d}_hits.cxi'
        vds_file     = f'{PREFIX}scratch/vds/r{run:04d}.cxi'
        sidecar_file = f'{cxi_file}.bg.h5'

        if args.merge:
            merge_sidecar_into_cxi(sidecar_file, cxi_file)
            continue

        dc = extra_data.open_run(proposal, run, data='all')
        ev = mcf.load_facility_data(dc, vds_file)

        back, n_used, frame_counts = compute_background_from_misses(
            vds_file, ev, max_trains=args.max_trains,
            k_sigma=args.k_sigma, nproc=args.nproc)
        back_counts = float(back.sum())
        print(f'{run=} {back_counts=:.3g} (from {n_used} frames)', file=sys.stderr)

        b = compute_weighting(ev, frame_counts, back_counts)

        write_sidecar(sidecar_file, back, b)
        print(f'wrote {sidecar_file}')

    print('add_background --merge done' if args.merge else 'add_background sidecar done')


if __name__ == '__main__':
    main()
