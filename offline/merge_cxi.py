"""
Merge per-run hit CXI files that share a common sample name into one CXI file.

Each per-run file written by make_cxi_file.py lives at
    {PREFIX}scratch/saved_hits/r{run:04d}_hits.cxi
and records its sample name at /entry_1/sample_1/name. This tool groups the
files by that name and concatenates all the per-shot quantities into a single
spec-compatible CXI file, preserving the layout of the inputs.

What is concatenated vs. shared
-------------------------------
Per-shot datasets (indexed by experiment_identifier) are concatenated in run
order:
    experiment_identifier, energy, pulse_energy, trainId, cellId, vds_index,
    score/<name>, electrospray/<param>, background_weighting, data
The union of the score columns and electrospray parameters across the inputs is
taken; a file missing a given column contributes NaN for its shots.

Per-pixel / per-module arrays are taken from the FIRST run (with a warning if a
later run differs):
    module_identifier, xyz_map, mask, distance, x/y_pixel_size
(xyz_map is the legacy per-pixel map and stays at the first run's geometry.)

The AGIPD panel geometry changes per run, so corner_position and basis_vectors
are written PER EVENT instead — each shot carries the geometry of its own run,
broadcast from that run's file:
    corner_position  (Nevents, NMODULES, 3)       axes experiment_identifier:module_identifier:coordinate
    basis_vectors    (Nevents, NMODULES, 2, 3)    axes experiment_identifier:module_identifier:dimension:coordinate
These are per-run-constant so they compress well; a run whose file lacks them
falls back to the first run's geometry.

Background (data_white) is merged by averaging
----------------------------------------------
The beamline background data_white differs per run, so the merged image is the
plain per-pixel mean of the runs' data_white. For a shot d from run r the
background actually subtracted downstream is b_d * W_r (weighting * data_white).
To keep that estimate unchanged on average after replacing W_r with the merged
W, the per-shot weighting is rescaled so the pixel-mean product is preserved:

    mean_pix(b'_d * W_merged) = mean_pix(b_d * W_r)
        =>  b'_d = b_d * mean_pix(W_r) / mean_pix(W_merged)

A per-shot `run` column is added so each shot can be traced back to its run.

The per-pixel `powder` (sum of the good-hit frames, written per run by
add_is_hit_cxi.py) is merged by summing the runs' powders.

The per-shot `score/hit_score_mask` column (sum_pix(mask * frame), written per
run by make_cxi_file.py) is concatenated like any other score column. The scalar
`score/hit_score_mask_data_white` (sum_pix(mask * data_white), written per run by
add_background_cxi.py) is merged by averaging across runs, matching the per-pixel
mean used for data_white itself.

Usage
-----
    python merge_cxi.py --list                 # list samples and their runs
    python merge_cxi.py "Sample name"          # merge one sample
    python merge_cxi.py --all                  # merge every sample group
    python merge_cxi.py --runs 12 13 17        # merge an explicit list of runs
    python merge_cxi.py "Sample" -o out.cxi    # explicit output path
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import h5py
from tqdm import tqdm

from constants import PREFIX

SAVED_HITS_DIR = f'{PREFIX}scratch/saved_hits'

# scalar score merged by averaging across runs (not a per-shot column)
SCALAR_SCORES = ('hit_score_mask_data_white',)

DET_PATH    = 'entry_1/instrument_1/detector_1'
SCORE_PATH  = f'{DET_PATH}/score'
ES_PATH     = 'entry_1/instrument_1/electrospray'

# per-module/per-pixel datasets copied from the first run (constant across shots).
# corner_position and basis_vectors are NOT here: they track the per-run AGIPD
# geometry and are written per-event instead (see scan_inputs / write_merged).
FIRST_RUN_DET = ('module_identifier', 'xyz_map', 'mask', 'distance',
                 'x_pixel_size', 'y_pixel_size', 'description')

GZ = dict(compression='gzip', compression_opts=1, shuffle=True)


def get_proposal_from_prefix():
    """Extract proposal number from EXP_PREFIX path, e.g. .../p010662/ -> 10662."""
    for part in PREFIX.rstrip('/').split('/'):
        if part.startswith('p') and part[1:].isdigit():
            return int(part[1:])
    raise RuntimeError(f'could not parse proposal number from PREFIX={PREFIX!r}')


def run_of(path):
    """Run number parsed from an r####_hits.cxi filename."""
    m = re.search(r'r(\d+)_hits\.cxi$', os.path.basename(path))
    return int(m.group(1)) if m else None


def sample_of(path):
    """Sample name stored in a hit CXI file, or None if unreadable."""
    try:
        with h5py.File(path, 'r') as f:
            if 'entry_1/sample_1/name' not in f:
                return None
            v = f['entry_1/sample_1/name'][()]
        return v.decode() if isinstance(v, bytes) else str(v)
    except OSError:
        return None


def group_by_sample(directory):
    """Map sample name -> sorted list of (run, path) over all hit CXI files."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(directory, 'r*_hits.cxi'))):
        run = run_of(path)
        if run is None:
            continue
        name = sample_of(path)
        if name is None:
            print(f'warning: skipping unreadable/sample-less file {path}')
            continue
        groups.setdefault(name, []).append((run, path))
    for name in groups:
        groups[name].sort()
    return groups


def safe_name(sample):
    """Filesystem-safe slug for a sample name."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', sample).strip('_') or 'sample'


# ---------------------------------------------------------------------------

def scan_inputs(files):
    """First pass: read per-shot 1D arrays and per-run data_white from every file.

    Returns a dict with the concatenated 1D columns, the merged data_white and
    rescaled background_weighting, the per-shot run column, total event count and
    the common frame_shape. The big `data` block is NOT read here; it is streamed
    later in copy_frames().
    """
    runs   = [run for run, _ in files]
    paths  = [path for _, path in files]

    nevents      = []
    frame_shape  = None
    start_times  = []

    # plain per-shot columns concatenated as-is
    plain_keys = ['experiment_identifier', 'trainId', 'cellId', 'vds_index']
    src_keys   = {'experiment_identifier': 'entry_1/experiment_identifier',
                  'trainId': f'{DET_PATH}/trainId',
                  'cellId':  f'{DET_PATH}/cellId',
                  'vds_index': f'{DET_PATH}/vds_index',
                  'energy': 'entry_1/instrument_1/source_1/energy',
                  'pulse_energy': 'entry_1/instrument_1/source_1/pulse_energy'}
    plain_keys += ['energy', 'pulse_energy']
    plain = {k: [] for k in plain_keys}

    # union columns (present in some files only): score/* and electrospray/*.
    # SCALAR_SCORES are per-file scalars (not per-shot) merged separately below.
    score_names = set()
    es_names    = set()
    for path in paths:
        with h5py.File(path, 'r') as f:
            if SCORE_PATH in f:
                score_names.update(k for k in f[SCORE_PATH].keys()
                                   if k not in SCALAR_SCORES)
            if ES_PATH in f:
                es_names.update(k for k in f[ES_PATH].keys()
                                if k != 'experiment_identifier'
                                and isinstance(f[f'{ES_PATH}/{k}'], h5py.Dataset))
    score = {n: [] for n in sorted(score_names)}
    es    = {n: [] for n in sorted(es_names)}

    # background pieces
    have_bg   = False
    W_runs    = []          # per-run data_white (or None)
    m_runs    = []          # per-run pixel-mean of data_white (or nan)
    bw_runs   = []          # per-run background_weighting (or None)

    # per-run scalar scores, averaged across runs (see SCALAR_SCORES)
    scalar_runs = {n: [] for n in SCALAR_SCORES}

    powder = None           # running per-pixel sum of every run's powder

    first_det = {}          # first-run per-module datasets (+ attrs)

    # per-run AGIPD panel geometry, broadcast to per-event below
    geom_runs  = {'corner_position': [], 'basis_vectors': []}
    geom_attrs = {}

    for i, path in enumerate(tqdm(paths, desc='scanning')):
        with h5py.File(path, 'r') as f:
            data = f[f'{DET_PATH}/data']
            n = data.shape[0]
            nevents.append(n)
            fs = data.shape[1:]
            if frame_shape is None:
                frame_shape = fs
            elif fs != frame_shape:
                raise RuntimeError(
                    f'frame shape {fs} in {path} != {frame_shape} from first run')

            st = f.get('entry_1/start_time')
            if st is not None:
                v = st[()]
                start_times.append(v.decode() if isinstance(v, bytes) else str(v))

            for k in plain_keys:
                plain[k].append(f[src_keys[k]][()])

            for n_, lst in score.items():
                p = f'{SCORE_PATH}/{n_}'
                lst.append(f[p][()] if p in f else np.full(n, np.nan, np.float32))
            for n_, lst in es.items():
                p = f'{ES_PATH}/{n_}'
                lst.append(f[p][()] if p in f else np.full(n, np.nan, np.float32))

            for n_, lst in scalar_runs.items():
                p = f'{SCORE_PATH}/{n_}'
                lst.append(float(f[p][()]) if p in f else np.nan)

            # powder: per-pixel sum across all runs' powders
            pp = f'{DET_PATH}/powder'
            if pp in f:
                w = f[pp][()].astype(np.float64)
                powder = w if powder is None else powder + w

            # first-run per-module arrays (and warn if mask differs later)
            if i == 0:
                for k in FIRST_RUN_DET:
                    p = f'{DET_PATH}/{k}'
                    if p in f:
                        first_det[k] = (f[p][()], dict(f[p].attrs))
            else:
                p = f'{DET_PATH}/mask'
                if 'mask' in first_det and p in f and \
                   not np.array_equal(f[p][()], first_det['mask'][0]):
                    print(f'warning: mask in run {runs[i]} differs from first '
                          f'run {runs[0]}; using first run')

            # per-run panel geometry (written per-event later); None if absent
            for k in ('corner_position', 'basis_vectors'):
                p = f'{DET_PATH}/{k}'
                if p in f:
                    geom_runs[k].append(f[p][()])
                    geom_attrs.setdefault(k, dict(f[p].attrs))
                else:
                    geom_runs[k].append(None)

            # background
            wp = f'{DET_PATH}/data_white'
            bp = f'{DET_PATH}/background_weighting'
            if wp in f:
                have_bg = True
                W = f[wp][()].astype(np.float64)
                W_runs.append(W)
                m_runs.append(float(np.nanmean(W)))
                bw_runs.append(f[bp][()].astype(np.float64) if bp in f
                               else np.full(n, np.nan))
            else:
                W_runs.append(None)
                m_runs.append(np.nan)
                bw_runs.append(np.full(n, np.nan) if bp not in f
                               else f[bp][()].astype(np.float64))

    nevents = np.asarray(nevents)
    total   = int(nevents.sum())

    # concatenate small columns
    cols = {k: np.concatenate(v) for k, v in plain.items()}
    cols_score = {k: np.concatenate(v) for k, v in score.items()}
    cols_es    = {k: np.concatenate(v) for k, v in es.items()}
    run_col = np.repeat(np.asarray(runs, np.uint16), nevents)

    # per-event panel geometry: broadcast each run's array to its events. A run
    # whose file lacks the geometry falls back to the first run that has it.
    geom_cols = {}
    for k, arrs in geom_runs.items():
        ref = next((a for a in arrs if a is not None), None)
        if ref is None:
            continue
        parts = []
        for i, a in enumerate(arrs):
            if a is None:
                print(f'warning: run {runs[i]} has no {k}; using fallback geometry')
                a = ref
            parts.append(np.broadcast_to(a[None], (nevents[i],) + a.shape))
        geom_cols[k] = np.concatenate(parts, axis=0)

    # merge background and rescale weighting
    data_white = None
    background_weighting = None
    if have_bg:
        stack = np.stack([W for W in W_runs if W is not None], axis=0)
        data_white = np.nanmean(stack, axis=0).astype(np.float32)
        m_merged = float(np.nanmean(data_white))
        bw_parts = []
        for m_r, bw in zip(m_runs, bw_runs):
            if np.isfinite(m_r) and m_merged > 0:
                bw_parts.append(bw * (m_r / m_merged))
            else:
                bw_parts.append(np.full(len(bw), np.nan))
        background_weighting = np.concatenate(bw_parts).astype(np.float32)

    # scalar scores: average across runs (skip if no run had the value)
    scalar_score = {}
    for n_, vals in scalar_runs.items():
        if np.any(np.isfinite(vals)):
            scalar_score[n_] = np.float32(np.nanmean(vals))

    return dict(
        total=total, frame_shape=frame_shape, paths=paths, nevents=nevents,
        start_time=min(start_times) if start_times else '',
        cols=cols, score=cols_score, es=cols_es, run=run_col,
        first_det=first_det, geom=geom_cols, geom_attrs=geom_attrs,
        data_white=data_white, background_weighting=background_weighting,
        scalar_score=scalar_score, powder=powder,
    )


def copy_frames(out_file, paths, nevents, frame_shape, block=256):
    """Stream the per-run `data` blocks into the merged dataset at running offset."""
    with h5py.File(out_file, 'a') as g:
        dset = g[f'{DET_PATH}/data']
        off = 0
        for path, n in zip(paths, tqdm(nevents, desc='copying frames')):
            with h5py.File(path, 'r') as f:
                src = f[f'{DET_PATH}/data']
                for s in range(0, n, block):
                    e = min(s + block, n)
                    dset[off + s:off + e] = src[s:e]
            off += n


def write_merged(out_file, sample, info, proposal):
    total       = info['total']
    frame_shape = info['frame_shape']
    expid       = info['cols']['experiment_identifier']

    with h5py.File(out_file, 'w') as f:
        f['cxi_version'] = 160

        entry = f.create_group('entry_1')
        entry.attrs['NX_class'] = 'NXentry'
        entry['start_time'] = info['start_time']
        entry['title'] = f'p{proposal} {sample} merged'
        ds = entry.create_dataset('experiment_identifier', data=expid)

        sgrp = entry.create_group('sample_1')
        sgrp.attrs['NX_class'] = 'NXsample'
        sgrp['name'] = sample

        instrument = entry.create_group('instrument_1')
        instrument.attrs['NX_class'] = 'NXinstrument'
        instrument['name'] = 'SPB'

        source = instrument.create_group('source_1')
        source.attrs['NX_class'] = 'NXsource'
        source['name'] = 'European XFEL SASE1'
        for k, units in (('energy', 'J'), ('pulse_energy', 'J')):
            d = source.create_dataset(k, data=info['cols'][k].astype(np.float32),
                                      **GZ)
            d.attrs['units'] = units
            d.attrs['axes']  = 'experiment_identifier'

        if info['es']:
            es = instrument.create_group('electrospray')
            es.attrs['NX_class'] = 'NXcollection'
            es['experiment_identifier'] = h5py.SoftLink(
                '/entry_1/experiment_identifier')
            for name, col in info['es'].items():
                d = es.create_dataset(name, data=col.astype(np.float32), **GZ)
                d.attrs['axes'] = 'experiment_identifier'

        detector = instrument.create_group('detector_1')
        detector.attrs['NX_class'] = 'NXdetector'

        # per-module arrays from the first run, attrs preserved
        for k, (val, attrs) in info['first_det'].items():
            chunks = frame_shape if k == 'mask' else None
            d = detector.create_dataset(k, data=val,
                                        **(dict(chunks=chunks, **GZ) if k == 'mask'
                                           else ({} if np.ndim(val) == 0 else GZ)))
            for ak, av in attrs.items():
                d.attrs[ak] = av

        # per-event panel geometry (each shot carries its run's geometry)
        geom_axes = {
            'corner_position': 'experiment_identifier:module_identifier:coordinate',
            'basis_vectors':
                'experiment_identifier:module_identifier:dimension:coordinate',
        }
        for k, col in info['geom'].items():
            d = detector.create_dataset(k, data=col.astype(np.float32),
                                        chunks=(1,) + col.shape[1:], **GZ)
            for ak, av in info['geom_attrs'].get(k, {}).items():
                d.attrs[ak] = av
            d.attrs['axes'] = geom_axes[k]

        detector['experiment_identifier'] = h5py.SoftLink(
            '/entry_1/experiment_identifier')
        for k, dt, axes in (('trainId', np.uint64, 'experiment_identifier'),
                            ('cellId',  np.uint16, 'experiment_identifier'),
                            ('vds_index', np.uint64, 'experiment_identifier'),
                            ('run', np.uint16, 'experiment_identifier')):
            src = info['run'] if k == 'run' else info['cols'][k]
            d = detector.create_dataset(k, data=src.astype(dt), **GZ)
            d.attrs['axes'] = axes

        if info['score'] or info['scalar_score']:
            sg = detector.create_group('score')
            for name, col in info['score'].items():
                d = sg.create_dataset(name, data=col.astype(np.float32), **GZ)
                d.attrs['axes'] = 'experiment_identifier'
            # per-file scalar scores averaged across runs (e.g. hit_score_mask_data_white)
            for name, val in info['scalar_score'].items():
                sg.create_dataset(name, data=val)

        if info['powder'] is not None:
            pw = detector.create_dataset('powder', data=info['powder'], **GZ)
            pw.attrs['axes']  = 'module_identifier:y:x'
            pw.attrs['units'] = 'counts'

        if info['data_white'] is not None:
            dw = detector.create_dataset('data_white', data=info['data_white'],
                                         **GZ)
            dw.attrs['axes']  = 'module_identifier:y:x'
            dw.attrs['units'] = 'counts'
            bw = detector.create_dataset('background_weighting',
                                         data=info['background_weighting'], **GZ)
            bw.attrs['axes'] = 'experiment_identifier'

        data = detector.create_dataset(
            'data', shape=(total,) + frame_shape, dtype=np.uint8,
            chunks=(1,) + frame_shape, **GZ)
        data.attrs['axes']   = 'experiment_identifier:module_identifier:y:x'
        data.attrs['signal'] = 1
        data.attrs['units']  = 'counts'

        dgrp = entry.create_group('data_1')
        dgrp.attrs['NX_class'] = 'NXdata'
        dgrp['data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')


def merge_sample(sample, files, output, proposal):
    runs = [r for r, _ in files]
    print(f'\nmerging sample {sample!r}: runs {runs} -> {output}')
    info = scan_inputs(files)
    print(f'  {len(files)} runs, {info["total"]} total shots, '
          f'frame {info["frame_shape"]}'
          f'{", background merged" if info["data_white"] is not None else ""}')
    write_merged(output, sample, info, proposal)
    copy_frames(output, info['paths'], info['nevents'], info['frame_shape'])
    print(f'  wrote {output}')


def main():
    p = argparse.ArgumentParser(
        description='Merge per-run hit CXI files that share a sample name.')
    p.add_argument('sample', nargs='?',
                   help='sample name to merge (omit with --list/--all)')
    p.add_argument('--list', action='store_true',
                   help='list available sample names and their runs, then exit')
    p.add_argument('--all', action='store_true',
                   help='merge every sample group found')
    p.add_argument('-r', '--runs', type=int, nargs='+', metavar='RUN',
                   help='merge this explicit list of run numbers instead of '
                        'selecting by sample name (the sample name of the first '
                        'run is used; a warning is printed if the runs disagree)')
    p.add_argument('-o', '--output', help='output CXI path (single sample only)')
    p.add_argument('-d', '--dir', default=SAVED_HITS_DIR,
                   help=f'directory of r####_hits.cxi files (default {SAVED_HITS_DIR})')
    args = p.parse_args()

    groups = group_by_sample(args.dir)
    if not groups:
        sys.exit(f'no r####_hits.cxi files found in {args.dir}')

    if args.list:
        for name in sorted(groups):
            runs = [r for r, _ in groups[name]]
            print(f'{name!r}: {len(runs)} runs {runs}')
        return

    proposal = get_proposal_from_prefix()

    if args.all:
        if args.output or args.runs or args.sample:
            sys.exit('--all cannot be combined with --output/--runs/sample')
        for name in sorted(groups):
            out = os.path.join(args.dir, f'{safe_name(name)}_merged.cxi')
            merge_sample(name, groups[name], out, proposal)
        return

    if args.runs:
        if args.sample:
            sys.exit('give a sample name OR --runs, not both')
        run_to_path = {r: p for files in groups.values() for r, p in files}
        missing = [r for r in args.runs if r not in run_to_path]
        if missing:
            sys.exit(f'no hit CXI file for run(s) {missing} in {args.dir}')
        # keep the user's order, drop duplicates
        seen, files = set(), []
        for r in args.runs:
            if r not in seen:
                seen.add(r)
                files.append((r, run_to_path[r]))
        names = {sample_of(p) for _, p in files}
        if len(names) > 1:
            print(f'warning: selected runs span multiple samples {sorted(names)}; '
                  f'using {sample_of(files[0][1])!r}')
        sample = sample_of(files[0][1])
        out = args.output or os.path.join(
            args.dir, f'{safe_name(sample)}_r{"-".join(str(r) for r, _ in files)}'
                      f'_merged.cxi')
        merge_sample(sample, files, out, proposal)
        return

    if not args.sample:
        sys.exit('give a sample name, or use --list / --all')
    if args.sample not in groups:
        sys.exit(f'sample {args.sample!r} not found; available: '
                 f'{sorted(groups)}')
    out = args.output or os.path.join(
        args.dir, f'{safe_name(args.sample)}_merged.cxi')
    merge_sample(args.sample, groups[args.sample], out, proposal)


if __name__ == '__main__':
    main()
