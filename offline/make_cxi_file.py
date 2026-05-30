"""
Write a CXI v1.6 file for the hits in a run.

All per-shot quantities are read from native EuXFEL Karabo sources via extra_data:
    hit_flag, miss_flag, hit_score, threshold mu/sig  <-  SPI_HITFINDER
    photons (energy per detector frame)               <-  LITFRM
    pulse_energy                                      <-  XGM (μJ -> J)
    photon energy / wavelength                        <-  undulator (keV -> J/m)
    trainId, cellId, pulseId per frame                <-  AGIPD module 0 CORR
Hit-finder pulses are aligned to AGIPD frames by (trainId, pulseId).

Layout (CXIDB spec, NeXus-compatible). Per-shot arrays are indexed by
experiment_identifier; per-pixel arrays by module_identifier:y:x.

    /
      cxi_version = 160                                            int (160 = v1.6)
      entry_1/                                                     NXentry
        experiment_identifier   (Nevents,)      string             p{proposal}_r{run}_t{train}_c{cell}
        start_time              scalar          string             ISO 8601 (extra_data)
        title                   scalar          string             "p{proposal} run {run}"
        sample_1/                                                  NXsample
          name                  scalar          string             from run_table.json
        instrument_1/                                              NXinstrument
          name                  scalar          string             "SPB"
          source_1/                                                NXsource
            name                scalar          string             "European XFEL SASE1"
            energy              (Nevents,)      float32  J         per-shot photon energy
            pulse_energy        (Nevents,)      float32  J         per-shot XGM pulse energy
          detector_1/                                              NXdetector
            description         scalar          string             "AGIPD 1M"
            distance            scalar          float    m         sample-to-detector (DET_DIST)
            x_pixel_size        scalar          float    m         from extra_geom
            y_pixel_size        scalar          float    m         from extra_geom
            corner_position     (NMODULES, 3)   float32  m         corner of pixel (0,0) per module
            basis_vectors       (NMODULES, 2, 3) float32 m         pixel-step vectors (ss, fs)
            module_identifier   (NMODULES,)     string             "AGIPD00"..."AGIPD15"
            xyz_map             (3, NMODULES, ss, fs) float32 m    legacy per-pixel position; z = DET_DIST
            trainId             (Nevents,)      uint64
            cellId              (Nevents,)      uint16
            vds_index           (Nevents,)      uint64             index into the per-run VDS
            mask                (NMODULES,ss,fs) uint32            CXI mask bits (0 = good)
            score/                                                 per-shot scalar group
              photon_counts?    (Nevents,)     float32             LITFRM energy per detector frame
              hit_score?        (Nevents,)     float32             SPI_HITFINDER hitscore
              hit_sigma?        (Nevents,)     float32             (hitscore - threshold.mu)/threshold.sig
            data                (Nevents, NMODULES, ss, fs) uint8  photon counts, per-cell-masked
            experiment_identifier -> /entry_1/experiment_identifier
        data_1/                                                    NXdata
          data -> /entry_1/instrument_1/detector_1/data

Added later by add_background_cxi.py:
    entry_1/instrument_1/detector_1/
      data_white              (NMODULES,ss,fs) float32 counts      beamline background (non-hit mean)
      background_weighting    (Nevents,)       float32             per-shot scale factor for data_white

Axes attributes used as dimension scales:
    energy, pulse_energy, trainId, cellId, vds_index, background_weighting
                                                  axes = "experiment_identifier"
    corner_position                               axes = "module_identifier:coordinate"
    basis_vectors                                 axes = "module_identifier:dimension:coordinate"
    mask, data_white                              axes = "module_identifier:y:x"
    data                                          axes = "experiment_identifier:module_identifier:y:x"
    score/photon_counts, hit_score, hit_sigma     axes = "experiment_identifier"
"""

import argparse
import json
import os
import sys
import multiprocessing as mp

import numpy as np
import h5py
import extra_data
import extra_geom
import scipy.constants as sc
from tqdm import tqdm

import common
from constants import PREFIX, DET_DIST

# Standard CXI path inside the per-run VDS file (as written by extra_data.write_virtual_cxi)
VDS_DATA_PATH = '/entry_1/instrument_1/detector_1/data'

# Karabo sources for native EuXFEL data. Edit these if facility config changes.
AGIPD_FRAME_COORD_SRC = 'SPB_DET_AGIPD1M-1/CORR/0CH0:output'      # module 0 (defines VDS frame coords)
HITFINDER_SRC         = 'SPB_DET_AGIPD1M-1/REDU/SPI_HITFINDER:output'
LITFRM_SRC            = 'SPB_IRU_AGIPD1M1/REDU/LITFRM:output'
XGM_SRC               = 'SPB_XTD9_XGM/XGM/DOOCS:output'           # downstream XGM near sample
UND_SRC               = 'SPB_XTD2_UND/DOOCS/ENERGY'               # undulator energy (control)
UND_ENERGY_KEY        = 'actualPosition'                          # keV per train

# CXI mask bits (cxi.h)
CXI_PIXEL_IS_BAD = 0x00000080


def probe_vds(vds_file):
    """Open the VDS once and return (frame_shape, nmodules, dtype)."""
    with h5py.File(vds_file) as g:
        shape = g[VDS_DATA_PATH].shape
        dtype = g[VDS_DATA_PATH].dtype
    frame_shape = shape[1:]            # everything past the events axis
    nmodules    = frame_shape[0]       # (NMODULES, ss, fs) layout
    return frame_shape, nmodules, dtype


def parse_args():
    p = argparse.ArgumentParser(description='Write hits to a spec-compliant CXI file')
    p.add_argument('run', type=int, help='Run number')
    p.add_argument('-m', '--mask', type=str,
                   help=f'Optional extra good-pixel mask in {PREFIX}scratch/det/, '
                        'AND-ed with the per-cell run mask before being written to '
                        '/entry_1/instrument_1/detector_1/mask. Frame data is never masked.')
    p.add_argument('-n', '--nproc', type=int, default=1, help='number of processes')
    p.add_argument('--max_frames', type=int, default=2048, help='maximum number of hits to include')
    args = p.parse_args()

    args.output_file = f'{PREFIX}scratch/saved_hits/r{args.run:04d}_hits.cxi'
    args.vds_file    = f'{PREFIX}scratch/vds/r{args.run:04d}.cxi'
    args.mask_file   = f'{PREFIX}scratch/det/{args.mask}' if args.mask else None
    args.geom_file   = common.get_geom(args.run)
    return args


def get_proposal_from_prefix():
    """Extract proposal number from EXP_PREFIX path, e.g. .../p010662/ -> 10662."""
    for part in PREFIX.rstrip('/').split('/'):
        if part.startswith('p') and part[1:].isdigit():
            return int(part[1:])
    raise RuntimeError(f'could not parse proposal number from PREFIX={PREFIX!r}')


def get_sample_name(run):
    """Read sample name from the autologger run table."""
    table_path = f'{PREFIX}scratch/log/run_table.json'
    with open(table_path) as f:
        table = json.load(f)
    for v in table.values():
        if isinstance(v, dict) and v.get('Run number') == run:
            return v['Sample']
    raise RuntimeError(f'sample name for run {run} not found in {table_path}')


def get_run_start_time(dc):
    """ISO 8601 start_time string from extra_data run metadata."""
    md = dc.run_metadata()
    # extra_data returns ISO 8601 strings like '2024-09-14T13:42:01+00:00'
    return md.get('creationDate') or md.get('beginning') or ''


# ----- Facility data via extra_data --------------------------------------------------

def _pair_key(train, pulse):
    """Pack (train, pulse) into a single int64 key for fast searchsorted lookup."""
    return (np.asarray(train, dtype=np.int64) << 32) | np.asarray(pulse, dtype=np.int64)


def _align_by_pair(target_train, target_pulse, source_train, source_pulse):
    """For each target (train, pulse), find the matching index in (source_train, source_pulse).

    Returns (matched_mask, source_index) both of length len(target_train); positions for
    unmatched rows are set to 0 but masked out by matched_mask.
    """
    target_key = _pair_key(target_train, target_pulse)
    source_key = _pair_key(source_train, source_pulse)
    order = np.argsort(source_key)
    sorted_keys = source_key[order]
    pos = np.searchsorted(sorted_keys, target_key)
    pos_clipped = np.clip(pos, 0, max(len(sorted_keys) - 1, 0))
    matched = (len(sorted_keys) > 0) & (sorted_keys[pos_clipped] == target_key)
    return matched, order[pos_clipped]


def _broadcast_per_train(values, value_trains, target_trains, fill=np.nan):
    """Broadcast a per-train scalar to a per-pulse array indexed by target_trains."""
    lookup = {int(t): float(v) for t, v in zip(value_trains, values)}
    out = np.full(len(target_trains), fill, dtype=float)
    for i, t in enumerate(target_trains):
        if int(t) in lookup:
            out[i] = lookup[int(t)]
    return out


def _pulse_resolved(kd):
    """Return (train_ids_per_pulse, values) as flat arrays of equal length for a
    pulse-resolved KeyData, regardless of whether the underlying layout is
    (Ntrains, Npulses) 2D or (Ntotal,) 1D."""
    data = kd.ndarray()
    tids = np.asarray(kd.train_id_coordinates())
    if data.ndim >= 2 and len(tids) == data.shape[0]:
        # (Ntrains, Npulses, ...) — broadcast train IDs along the pulse axis
        Ntrains = data.shape[0]
        Npulses = int(np.prod(data.shape[1:]))
        per_pulse_tids = np.broadcast_to(tids[:, None],
                                         (Ntrains, Npulses)).ravel()
        return per_pulse_tids, data.reshape(Ntrains * Npulses)
    if data.ndim == 1 and len(tids) == len(data):
        return tids, data
    if data.ndim == 1 and len(tids) != len(data):
        # 1D flat with one trainId per train — expand via per-train data counts
        counts = np.asarray(kd.data_counts(labelled=False)).astype(np.int64)
        if counts.sum() != len(data):
            raise RuntimeError(
                f'data_counts sum {counts.sum()} != data len {len(data)} for {kd.source}/{kd.key}')
        return np.repeat(tids, counts), data
    raise RuntimeError(
        f'unexpected layout: data shape {data.shape}, train_id_coordinates shape {tids.shape}')


def load_facility_data(dc, vds_file):
    """Pull per-shot quantities from native EuXFEL sources via extra_data, aligned to
    VDS frame index.

    Returns a dict keyed by VDS frame index k:
        trainId, cellId, pulseId          (Nframes,)
        is_hit, is_miss                   (Nframes,) bool
        hit_score, hit_sigma              (Nframes,) float32, NaN if unmatched
        photons                           (Nframes,) float32, NaN if LITFRM missing
        pulse_energy                      (Nframes,) float32 J, NaN if XGM missing
        wavelength                        (Nframes,) float32 m, NaN if undulator missing
    """
    # 1) AGIPD module 0 defines the (trainId, pulseId, cellId) of every VDS frame.
    agipd = dc[AGIPD_FRAME_COORD_SRC]
    a_train = agipd['image.trainId'].ndarray().ravel().astype(np.int64)
    a_pulse = agipd['image.pulseId'].ndarray().ravel().astype(np.int64)
    a_cell  = agipd['image.cellId'].ndarray().ravel().astype(np.int64)
    Nframes = len(a_train)

    # Cross-check against the VDS
    with h5py.File(vds_file) as g:
        vds_train = g['/entry_1/trainId'][:]
        vds_cell  = g['/entry_1/cellId'][:, 0]
    assert len(vds_train) == Nframes, \
        f'VDS frame count {len(vds_train)} disagrees with AGIPD module 0 {Nframes}'
    assert np.array_equal(vds_train, a_train), 'VDS trainId disagrees with AGIPD module 0'
    assert np.array_equal(vds_cell,  a_cell),  'VDS cellId disagrees with AGIPD module 0'

    out = dict(
        trainId=a_train.astype(np.uint64),
        cellId=a_cell.astype(np.uint16),
        pulseId=a_pulse.astype(np.uint64),
    )

    # 2) Hit info from the facility SPI hit finder.
    hf = dc[HITFINDER_SRC]
    hf_train, hf_pulse = _pulse_resolved(hf['data.pulseId'])
    matched, src_idx = _align_by_pair(a_train, a_pulse, hf_train, hf_pulse)
    n_missing = int((~matched).sum())
    if n_missing:
        print(f'WARNING: {n_missing}/{Nframes} frames have no hitfinder entry')

    def scatter(kd, fill, dtype):
        _, vals = _pulse_resolved(kd)
        v = np.full(Nframes, fill, dtype=dtype)
        v[matched] = vals[src_idx[matched]]
        return v

    out['is_hit']    = scatter(hf['data.hitFlag'],  False, bool)
    out['is_miss']   = scatter(hf['data.missFlag'], False, bool)
    out['hit_score'] = scatter(hf['data.hitscore'], np.nan, np.float32)

    # Threshold mu/sig are train-resolved; broadcast to per-pulse and compute hit_sigma.
    mu     = hf['threshold.mu'].ndarray().ravel()
    sig    = hf['threshold.sig'].ndarray().ravel()
    mu_tid = hf['threshold.mu'].train_id_coordinates()
    sig_tid = hf['threshold.sig'].train_id_coordinates()
    mu_pp  = _broadcast_per_train(mu,  mu_tid,  a_train)
    sig_pp = _broadcast_per_train(sig, sig_tid, a_train)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = (out['hit_score'] - mu_pp) / sig_pp
    sigma[~np.isfinite(sigma)] = np.nan
    out['hit_sigma'] = sigma.astype(np.float32)

    # 3) Photon counts proxy from LITFRM (energy per detector pulse)
    out['photons'] = np.full(Nframes, np.nan, dtype=np.float32)
    if LITFRM_SRC in dc.all_sources:
        lf = dc[LITFRM_SRC]
        lf_train, lf_pulse = _pulse_resolved(lf['data.detectorPulseId'])
        m, idx = _align_by_pair(a_train, a_pulse, lf_train, lf_pulse)
        _, vals = _pulse_resolved(lf['data.energyPerFrame'])
        out['photons'][m] = vals[idx[m]].astype(np.float32)
    else:
        print(f'WARNING: {LITFRM_SRC} not in run; photon_counts column will be NaN')

    # 4) Per-pulse XGM pulse energy (J). data.intensityTD is typically μJ.
    out['pulse_energy'] = np.full(Nframes, np.nan, dtype=np.float32)
    if XGM_SRC in dc.all_sources:
        xgm = dc[XGM_SRC]
        # data.intensityTD is (Ntrains, Npulses_max) and is padded; we index by xgmPulseId-like
        # tables. The per-pulse train arrives as ndarray() (Ntrains, Np). To align with AGIPD,
        # broadcast intensityTD[train_index, pulseId] via lookup.
        xgm_data = xgm['data.intensityTD'].ndarray()                # (Ntrains, Np_max) μJ
        xgm_tid  = xgm['data.intensityTD'].train_id_coordinates()
        train_to_row = {int(t): i for i, t in enumerate(xgm_tid)}
        for k in range(Nframes):
            row = train_to_row.get(int(a_train[k]))
            if row is None: continue
            p = int(a_pulse[k])
            if 0 <= p < xgm_data.shape[1]:
                out['pulse_energy'][k] = xgm_data[row, p] * 1e-6   # μJ → J
    else:
        print(f'WARNING: {XGM_SRC} not in run; pulse_energy column will be NaN')

    # 5) Photon wavelength from undulator (control source, per-train scalar in keV).
    # sc.e converts eV → J, so multiply by 1e3 to go from keV → J.
    out['wavelength'] = np.full(Nframes, np.nan, dtype=np.float32)
    if UND_SRC in dc.all_sources:
        und = dc[UND_SRC, UND_ENERGY_KEY]
        keV     = und.ndarray()
        keV_tid = und.train_id_coordinates()
        lam_per_pulse = np.empty(Nframes, dtype=np.float64)
        train_to_lam = {int(t): sc.h * sc.c / (float(e) * 1e3 * sc.e)
                        for t, e in zip(keV_tid, keV) if e > 0}
        for k in range(Nframes):
            lam_per_pulse[k] = train_to_lam.get(int(a_train[k]), np.nan)
        out['wavelength'] = lam_per_pulse.astype(np.float32)
    else:
        print(f'WARNING: {UND_SRC} not in run; wavelength/photon_energy will be NaN')

    return out


def build_detector_mask(vds_file, extra_mask_file, frame_shape):
    """Derive a per-pixel mask from the AGIPD calibration mask shipped in the VDS
    (/entry_1/instrument_1/detector_1/mask, written by extra_data.write_virtual_cxi).

    For each unique cellId we sample one frame's mask — calibration constants are
    per memory cell, so this captures the per-cell variation without scanning all
    frames. A pixel is treated as 'good' only if its mask is 0 (no bad bits set)
    in *every* sampled cell. Optional extra mask is AND-ed in. Returned only for
    writing to /detector_1/mask — frames themselves are written raw."""
    mask_path = '/entry_1/instrument_1/detector_1/mask'
    with h5py.File(vds_file) as g:
        if mask_path not in g:
            raise RuntimeError(f'{vds_file} has no {mask_path}; rebuild the VDS '
                               'with extra_data.write_virtual_cxi so the AGIPD '
                               'calibration mask is available.')
        cell_vds = g['/entry_1/cellId'][:, 0]
        mask_dset = g[mask_path]

        _, first_idx = np.unique(cell_vds, return_index=True)
        good = np.ones(frame_shape, dtype=bool)
        for idx in first_idx:
            good &= (np.squeeze(mask_dset[int(idx)]) == 0)
        ncells = len(first_idx)

    if extra_mask_file is not None:
        with h5py.File(extra_mask_file) as f:
            good &= f['entry_1/good_pixels'][()].astype(bool)
    return good, ncells


def geometry_to_cxi(geom_file):
    """Compute CXI-spec corner_position, basis_vectors, module_identifier, pixel
    size, and a legacy xyz_map per-pixel position array from a CrystFEL geom."""
    geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom_file)
    # shape (modules, ss, fs, 3) in metres, pixel centres
    pos = geom.get_pixel_positions()
    nmodules = pos.shape[0]

    basis_vectors   = np.empty((nmodules, 2, 3), dtype=np.float32)
    corner_position = np.empty((nmodules, 3),    dtype=np.float32)
    # basis_vectors[mod, 0, :] = step along the first (ss) data dimension
    # basis_vectors[mod, 1, :] = step along the second (fs) data dimension
    basis_vectors[:, 0, :] = pos[:, 1, 0, :] - pos[:, 0, 0, :]
    basis_vectors[:, 1, :] = pos[:, 0, 1, :] - pos[:, 0, 0, :]
    # spec: corner_position is the *corner* of pixel (0,0), not the centre
    corner_position[:] = (pos[:, 0, 0, :]
                          - 0.5 * basis_vectors[:, 0, :]
                          - 0.5 * basis_vectors[:, 1, :])
    # offset whole detector to live at z = DET_DIST (geom z is module-relative)
    corner_position[:, 2] += DET_DIST

    # Legacy per-pixel xyz map: (3, NMODULES, ss, fs); z overwritten with DET_DIST.
    # Not part of the CXI spec; kept for downstream code that imports it.
    xyz_map = np.transpose(pos, (3, 0, 1, 2)).astype(np.float32)
    xyz_map[2] = DET_DIST

    module_identifier = np.array([f'AGIPD{m:02d}' for m in range(nmodules)],
                                 dtype=h5py.string_dtype())
    return (corner_position, basis_vectors, module_identifier,
            float(geom.pixel_size), xyz_map)


def write_initial_file(args, ev, indices, proposal, start_time, sample_name,
                       corner_position, basis_vectors, module_identifier,
                       pixel_size, xyz_map, detector_good_pixels, frame_shape):
    Nevents = len(indices)
    photon_energy = (sc.h * sc.c / ev['wavelength']).astype(np.float32)

    # spec-compliant experiment identifiers: one unique string per shot
    expid = np.array(
        [f'p{proposal:06d}_r{args.run:04d}_t{int(ev["trainId"][i])}_c{int(ev["cellId"][i])}'
         for i in indices],
        dtype=h5py.string_dtype(),
    )

    # 32-bit CXI mask: 0 = good, CXI_PIXEL_IS_BAD on masked pixels
    mask = np.where(detector_good_pixels, 0, CXI_PIXEL_IS_BAD).astype(np.uint32)

    # per-shot scores written as 1D arrays under a `score` group; skip entries
    # that are entirely NaN
    score_columns = {}
    for name, src_key in (('photon_counts', 'photons'),
                          ('hit_score',     'hit_score'),
                          ('hit_sigma',     'hit_sigma')):
        if src_key not in ev:
            continue
        col = ev[src_key][indices].astype(np.float32)
        if np.all(np.isnan(col)):
            continue
        score_columns[name] = col

    gz = dict(compression='gzip', compression_opts=1, shuffle=True)

    with h5py.File(args.output_file, 'w') as f:
        f['cxi_version'] = 160

        entry = f.create_group('entry_1')
        entry.attrs['NX_class'] = 'NXentry'
        entry['start_time'] = start_time
        entry['title'] = f'p{proposal} run {args.run}'
        entry.create_dataset('experiment_identifier', data=expid)

        sample = entry.create_group('sample_1')
        sample.attrs['NX_class'] = 'NXsample'
        sample['name'] = sample_name

        instrument = entry.create_group('instrument_1')
        instrument.attrs['NX_class'] = 'NXinstrument'
        instrument['name'] = 'SPB'

        source = instrument.create_group('source_1')
        source.attrs['NX_class'] = 'NXsource'
        source['name'] = 'European XFEL SASE1'
        e = source.create_dataset('energy', data=photon_energy[indices], **gz)
        e.attrs['units'] = 'J'
        e.attrs['axes'] = 'experiment_identifier'
        pe = source.create_dataset('pulse_energy',
                                   data=ev['pulse_energy'][indices].astype(np.float32),
                                   **gz)
        pe.attrs['units'] = 'J'
        pe.attrs['axes'] = 'experiment_identifier'

        detector = instrument.create_group('detector_1')
        detector.attrs['NX_class'] = 'NXdetector'
        detector['description'] = 'AGIPD 1M'

        for name, val in (('distance', float(DET_DIST)),
                          ('x_pixel_size', pixel_size),
                          ('y_pixel_size', pixel_size)):
            ds = detector.create_dataset(name, data=val)
            ds.attrs['units'] = 'm'

        cp = detector.create_dataset('corner_position', data=corner_position)
        cp.attrs['units'] = 'm'
        cp.attrs['axes']  = 'module_identifier:coordinate'

        bv = detector.create_dataset('basis_vectors', data=basis_vectors)
        bv.attrs['units'] = 'm'
        bv.attrs['axes']  = 'module_identifier:dimension:coordinate'

        detector.create_dataset('module_identifier', data=module_identifier)

        # Legacy per-pixel position map (not part of the CXI spec). Z is forced to
        # DET_DIST. Kept so older downstream code that expects xyz_map keeps working.
        xyz_ds = detector.create_dataset('xyz_map', data=xyz_map, **gz)
        xyz_ds.attrs['units'] = 'm'
        xyz_ds.attrs['axes']  = 'coordinate:module_identifier:y:x'

        # per-shot identifiers used as dimension scale
        detector['experiment_identifier'] = h5py.SoftLink('/entry_1/experiment_identifier')
        detector.create_dataset('trainId',
                                data=ev['trainId'][indices].astype(np.uint64), **gz)
        detector.create_dataset('cellId',
                                data=ev['cellId'][indices].astype(np.uint16), **gz)
        detector.create_dataset('vds_index', data=indices.astype(np.uint64), **gz)

        m = detector.create_dataset('mask', data=mask,
                                    chunks=frame_shape, **gz)
        m.attrs['axes'] = 'module_identifier:y:x'

        if score_columns:
            score_grp = detector.create_group('score')
            for name, col in score_columns.items():
                ds = score_grp.create_dataset(name, data=col, **gz)
                ds.attrs['axes'] = 'experiment_identifier'

        data = detector.create_dataset(
            'data',
            shape=(Nevents,) + frame_shape,
            dtype=np.uint8,
            chunks=(1,) + frame_shape,
            **gz,
        )
        data.attrs['axes']   = 'experiment_identifier:module_identifier:y:x'
        data.attrs['signal'] = 1
        data.attrs['units']  = 'counts'

        data_group = entry.create_group('data_1')
        data_group.attrs['NX_class'] = 'NXdata'
        data_group['data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')


def write_frames(args, ev, indices, frame_shape, vds_dtype):
    """Fill detector_1/data with raw frames from the VDS (no masking applied).
    Parallelised over ranks; HDF5 writes serialised through a lock."""
    Nevents = len(indices)
    size = max(1, args.nproc)
    events_rank = np.linspace(0, Nevents, size + 1).astype(int)

    with h5py.File(args.vds_file) as g:
        cell_vds  = g['entry_1/cellId'][:, 0]
        train_vds = g['entry_1/trainId'][()]

    # confirm extra_data and VDS indexing agree before we read frames in workers
    assert np.array_equal(cell_vds[indices], ev['cellId'][indices])
    assert np.array_equal(train_vds[indices], ev['trainId'][indices])

    def worker(rank, lock):
        my_idx = indices[events_rank[rank]:events_rank[rank + 1]]
        buf = np.empty((len(my_idx),) + frame_shape, dtype=vds_dtype)

        with h5py.File(args.vds_file) as g:
            data = g[VDS_DATA_PATH]
            it = tqdm(range(len(my_idx)), desc=f'rank {rank} reading VDS') \
                 if rank == 0 else range(len(my_idx))
            for i in it:
                buf[i] = np.squeeze(data[my_idx[i]])

        hi = np.iinfo(np.uint8).max
        with lock, h5py.File(args.output_file, 'a') as f:
            dset = f['entry_1/instrument_1/detector_1/data']
            for i in range(len(my_idx)):
                dset[events_rank[rank] + i] = np.clip(buf[i], 0, hi)

    lock = mp.Lock()
    jobs = [mp.Process(target=worker, args=(r, lock)) for r in range(size)]
    for j in jobs: j.start()
    for j in jobs: j.join()


def main():
    args = parse_args()
    proposal = get_proposal_from_prefix()
    sample_name = get_sample_name(args.run)
    print(f'sample: {sample_name}')

    print(f'opening run via extra_data: p{proposal}, r{args.run}')
    dc = extra_data.open_run(proposal, args.run, data='all')
    start_time = get_run_start_time(dc)
    print(f'start_time: {start_time}')

    frame_shape, nmodules, vds_dtype = probe_vds(args.vds_file)
    print(f'VDS frame shape {frame_shape}, {nmodules} modules, dtype {vds_dtype}')

    print(f'deriving per-pixel mask from VDS calibration mask {args.vds_file}')
    detector_good, ncells = build_detector_mask(
        args.vds_file, args.mask_file, frame_shape)
    masked_frac = 100 * np.sum(~detector_good) / detector_good.size
    print(f'{masked_frac:.2f}% pixels masked (sampled across {ncells} cells)')

    (corner_position, basis_vectors, module_identifier,
     pixel_size, xyz_map) = geometry_to_cxi(args.geom_file)
    assert basis_vectors.shape[0] == nmodules, \
        f'geom has {basis_vectors.shape[0]} modules but VDS has {nmodules}'

    print(f'loading hit selection from {HITFINDER_SRC}')
    ev = load_facility_data(dc, args.vds_file)
    indices = np.where(ev['is_hit'])[0]
    print(f'found {len(indices)} hits')
    if args.max_frames is not None and len(indices) > args.max_frames:
        # Safety cap: a misconfigured hit finder can flag every frame, which
        # would explode the CXI size and runtime. Keep the first N hits in
        # acquisition order (deterministic, no RNG).
        print(f'WARNING: capping to first {args.max_frames} hits '
              f'(was {len(indices)}); pass --max_frames to override.')
        indices = indices[:args.max_frames]
    sys.stdout.flush()

    print(f'initialising {args.output_file}')
    write_initial_file(args, ev, indices, proposal, start_time, sample_name,
                       corner_position, basis_vectors, module_identifier,
                       pixel_size, xyz_map, detector_good, frame_shape)

    write_frames(args, ev, indices, frame_shape, vds_dtype)
    print('Done')


if __name__ == '__main__':
    main()
