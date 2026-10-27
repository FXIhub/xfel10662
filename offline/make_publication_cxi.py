"""
Write a CXI v1.6 file for publication with every hit of one sample, from the
per-run hit files, their metadata sidecars and the sample-conditions table:

    {PREFIX}scratch/saved_hits/r{run:04d}_hits.cxi           make_cxi_file.py + add_background_cxi.py
                                                            + classification, sizing, fluence, is_hit
    {PREFIX}scratch/saved_hits/r{run:04d}_hits.cxi.meta.h5   get_event_metadata.py
    ery_sample_conditions.csv                               --conditions

    python make_publication_cxi.py --conditions ery_sample_conditions.csv
    python make_publication_cxi.py --sample Erythrocruorin --conditions ery_sample_conditions.csv -o out.cxi

Every event of every run is included (the per-run files hold the facility hit
finder's hits); score/is_hit marks the analysis "good hits". The layout follows
the p7927 publication file (xfel7927/offline/make_publication_cxi.py). N is the
number of events, Nb the number of runs (one background per run).

    /
      cxi_version = 160
      entry_1/                                                     NXentry
        experiment_identifier   (N,)            string             p010662_r{run:04d}_t{trainId}_c{cellId}
        title, experiment_description, program_name, start_time, end_time
        sample_1/                                                  NXsample
          name                  scalar          string
          antifreeze_concentration (N,)         float32  %         \
          concentration         (N,)            float32  uM         | from --conditions
          calcium_concentration (N,)            float32  uM         | (merge_cxi.CONDITION_COLS),
          conditions_index      (N,)            int16              | NaN / -1 = not recorded
          conditions_table      scalar          string             /
        instrument_1/                                              NXinstrument
          source_1/                                                NXsource
            energy              (N,)            float32  J         photon energy, undulator (per train)
            pulse_energy        (N,)            float32  J         SPB_XTD9 XGM energy of this frame's pulse
            pulse_energy_sigma  (N,)            float32  J
            fluence_sizing      (N,)            float64  photons/m^2  from the sizing fit (run_emc_fluence.py),
                                                                   -1 = not sized
          xgm_1/, xgm_2/                                           SPB_XTD9 / SA1_XTD2 XGM: intensity(_sigma),
                                                                   x, y (_sigma) of this frame's pulse
          attenuator_1/, attenuator_2/                             SA1_XTD2 / SPB_XTD9 transmission (per train)
          electrospray/                                            injector, per train: gas flows, HV, motors,
                                                                   chamber_pressure; sample_backing_pressure
                                                                   from --conditions
          detector_1/                                              NXdetector
            description, distance, x_pixel_size, y_pixel_size
            module_identifier   (128,)          string             "AGIPD{mm}T{t}", tile t of module mm
            corner_position     (128, 3)        float32  m         corner of pixel (0, 0) of each tile
            basis_vectors       (128, 2, 3)     float32  m         pixel steps along y (slow) and x (fast)
            xyz_map             (3, 128, 64, 128) float32 m        pixel centres, same geometry
            quadrant            (4,)            string
            mask                (128, 64, 128)  uint32             CXI mask bits, see bit_definitions
            data                (N, 128, 64, 128) uint8  counts    photons per pixel (per-cell masked)
            powder              (128, 64, 128)  float32  counts    mean of data over all events
            start_time, run, trainId, cellId, pulseId, vds_index (N,)
            data_white          (Nb, 128, 64, 128) float32 counts  per-run background (mean of non-hit frames)
            background_run      (Nb,)           uint16
            background_index    (N,)            uint32             index into data_white
            background_weighting (N,)           float32            background = weighting * data_white[index]
            score/                                                 per-event scores, see descriptions
            motors/                                                AGIPD quadrant motors and z stage (per train)
            note_1/                                                geometry provenance
        result_1/                                                  EMC classification (merge_cxi.py layout)
        result_2/                                                  EMC sizing: data (scale x,y,z), w_d, ...
        data_1/                                                    NXdata, data -> detector_1/data
        process_1/                                                 NXprocess (program, version, date, command)

Detector tiles
    Each AGIPD module (512 x 128 pixels) is 8 tiles of 64 x 128 pixels separated
    by gaps, so one corner/basis per module cannot place every pixel. Tile
    "AGIPD{mm}T{t}" (index 8 * mm + t) is rows 64t..64t+63 of module mm; any
    per-pixel array reshapes (C order) to the facility layout (16, 512, 128).
    Pixel centre of (tile m, row i, column j):
        corner_position[m] + (i + 0.5) * basis_vectors[m, 0] + (j + 0.5) * basis_vectors[m, 1]
    The geometry is that of the per-run files (geom/r{run}.geom: the reference
    geometry moved to the recorded quadrant motor positions, z from the detector
    stage), identical for all runs here to within 0.01 pixel (checked).
"""
import argparse
import datetime
import os
import subprocess
import sys

import numpy as np
import h5py
import extra_geom
from tqdm import tqdm

from constants import PREFIX
import merge_cxi
from add_is_hit_cxi import is_good_hit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_GEOM = 'geom/agipd_p10662_pw_70cm_from_260505.geom'

DET = 'entry_1/instrument_1/detector_1'
GZ = dict(compression='gzip', compression_opts=1, shuffle=True)

NMODULES, NTILES = 16, 8
TILE_SHAPE = (64, 128)
PANEL_SHAPE = (NMODULES * NTILES,) + TILE_SHAPE

CXI_PIXEL_IS_BAD = 0x00000080

TIMESTAMP_DESCRIPTION = ('approximate train time (ISO 8601, +-1 s): run offset + 0.1 s * trainId, the '
                         'offset being the median over the train timestamps recorded in the run')

# descriptions of the per-run score columns carried through
SCORE_DESCRIPTIONS = {
    'hit_score': 'facility SPI hit finder hitscore (SPB_DET_AGIPD1M-1/REDU/SPI_HITFINDER data.hitscore)',
    'hit_sigma': '(hit_score - threshold.mu) / threshold.sig of the facility hit finder (per train)',
    'hit_score_mask': 'photons in the hit-finding good-pixel mask (scratch/det/hit_finding_mask.h5)',
}

# "good hit" criterion (add_is_hit_cxi.is_good_hit), as run by DAMNIT (context.py good_hits);
# recomputed here for every event so all runs use the same thresholds
GOOD_HIT = dict(hit_score_threshold=500., size_min=0.7, size_max=1.2, good_classes=(0, 1, 2, 3, 4, 5, 6))
IS_HIT_DESCRIPTION = ('analysis "good hit" (add_is_hit_cxi.is_good_hit): EMC class (result_1/data) in '
                      f'{list(GOOD_HIT["good_classes"])}, all three sizing scale factors (result_2/data) in '
                      f'[{GOOD_HIT["size_min"]}, {GOOD_HIT["size_max"]}] and hit_score_mask > '
                      f'{GOOD_HIT["hit_score_threshold"]:g}')
# per-run score columns replaced here: photon_counts held LITFRM energyPerFrame (a
# pulse energy), so it is recomputed from the frames instead; is_hit is recomputed
SCORES_DROPPED = ('photon_counts', 'is_hit')


def to_tiles(a):
    """(..., 16, 512, 128) -> (..., 128, 64, 128)"""
    return a.reshape(a.shape[:-3] + PANEL_SHAPE)


def git_version():
    try:
        return subprocess.check_output(['git', '-C', ROOT, 'describe', '--always', '--dirty'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unknown'


def link_expid(group):
    if 'experiment_identifier' not in group:
        group['experiment_identifier'] = h5py.SoftLink('/entry_1/experiment_identifier')


def per_event(group, name, data, units=None, **attrs):
    ds = group.create_dataset(name, data=data, **GZ)
    ds.attrs['axes'] = 'experiment_identifier'
    if units:
        ds.attrs['units'] = units
    for k, v in attrs.items():
        ds.attrs[k] = v
    link_expid(group)
    return ds


def load_metadata(runs, nevents):
    """concatenate the per-run metadata sidecars in run order; union of columns,
    NaN / -1 where a run lacks one. Returns (columns, attrs)."""
    data, attrs = {}, {}
    total = int(np.sum(nevents))
    offsets = np.cumsum(nevents) - nevents
    for run, n, off in zip(runs, nevents, offsets):
        fnam = f'{PREFIX}scratch/saved_hits/r{run:04d}_hits.cxi.meta.h5'
        with h5py.File(fnam, 'r') as f:
            if f['pulseId'].shape[0] != n:
                sys.exit(f'{fnam} has {f["pulseId"].shape[0]} events, the CXI has {n}')

            def collect(name, ds):
                if not isinstance(ds, h5py.Dataset):
                    return
                if name not in data:
                    if ds.dtype.kind == 'O':
                        data[name] = np.full(total, '', dtype=object)
                    else:
                        fill = np.nan if ds.dtype.kind == 'f' else -1
                        data[name] = np.full(total, fill, dtype=ds.dtype)
                    attrs[name] = dict(ds.attrs)
                v = ds.asstr()[()] if ds.dtype.kind == 'O' else ds[()]
                data[name][off:off + n] = v
            f.visititems(collect)
    return data, attrs


def copy_meta(group, name, meta, meta_attrs, key, **attrs):
    """copy a metadata column, keeping its units/source/key attributes"""
    if key not in meta:
        print(f'warning: {key} not in the metadata, {group.name}/{name} skipped')
        return
    units = str(meta_attrs[key].get('units', ''))
    known = units and not units.startswith('unknown')
    out = per_event(group, name, meta[key], units if known else None, **attrs)
    if units.startswith('unknown'):
        out.attrs['units_note'] = units + ' (not recorded in the facility data)'
    for a in ('source', 'key'):
        if meta_attrs[key].get(a):
            out.attrs[a] = meta_attrs[key][a]


def load_geometry(run, xyz_ref):
    """per-tile corner_position, basis_vectors, xyz_map from geom/r{run}.geom,
    checked to be exact per tile and to reproduce the per-run files' xyz_map"""
    fnam = os.path.join(ROOT, f'geom/r{run:04d}.geom')
    geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(fnam)
    pos = geom.get_pixel_positions()                        # (16, 512, 128, 3)
    err = np.abs(np.moveaxis(pos, -1, 0) - xyz_ref).max() / geom.pixel_size
    assert err < 1e-3, f'{fnam} differs from the per-run xyz_map by {err} pixels'
    pos = pos.reshape(PANEL_SHAPE + (3,))

    basis = np.empty((pos.shape[0], 2, 3), dtype=np.float64)
    basis[:, 0] = pos[:, 1, 0] - pos[:, 0, 0]     # slow scan (y)
    basis[:, 1] = pos[:, 0, 1] - pos[:, 0, 0]     # fast scan (x)
    corner = pos[:, 0, 0] - 0.5 * basis[:, 0] - 0.5 * basis[:, 1]

    i, j = np.mgrid[:TILE_SHAPE[0], :TILE_SHAPE[1]] + 0.5
    model = corner[:, None, None] + i[..., None] * basis[:, None, None, 0] + j[..., None] * basis[:, None, None, 1]
    err = np.abs(model - pos).max() / geom.pixel_size
    assert err < 1e-3, f'per-tile geometry is not exact ({err} pixels)'

    xyz = np.transpose(pos, (3, 0, 1, 2)).astype(np.float32)
    return corner.astype(np.float32), basis.astype(np.float32), xyz, float(geom.pixel_size), fnam


def command_line():
    args = [os.path.basename(a) if os.path.isabs(a) else a for a in sys.argv[1:]]
    return ' '.join([os.path.basename(sys.argv[0])] + args)


def write_beam_and_injector(instrument, meta, ma, fluence, fluence_attrs):
    source = instrument.create_group('source_1')
    source.attrs['NX_class'] = 'NXsource'
    source['name'] = 'European XFEL SASE1'
    link_expid(source)
    copy_meta(source, 'energy', meta, ma, 'train/undulator_energy',
              description='photon energy, undulator setting (per train)')
    pe = per_event(source, 'pulse_energy', meta['pulse/XGM_SPB_XTD9_intensityTD'].astype(np.float32), 'J',
                   description='XGM (SPB_XTD9) pulse energy of the pulse that produced this frame '
                               '(XGM slot via LITFRM xgmPulseId at the frame\'s pulse rank)')
    pe.attrs.update(source=ma['pulse/XGM_SPB_XTD9_intensityTD']['source'], key='data.intensityTD')
    copy_meta(source, 'pulse_energy_sigma', meta, ma, 'pulse/XGM_SPB_XTD9_intensitySigmaTD')
    if fluence is not None:
        per_event(source, 'fluence_sizing', fluence, 'photons/m^2',
                  description='incident fluence on the particle from the sizing fit '
                              '(run_emc_fluence.py: I0 = w_d / (tau r0^2 C_max I_max)); -1 = not sized',
                  **fluence_attrs)

    for i, (name, desc) in enumerate((('XGM_SPB_XTD9', 'SPB_XTD9_XGM, downstream gas monitor'),
                                     ('XGM_SA1_XTD2', 'SA1_XTD2_XGM, upstream gas monitor')), 1):
        xgm = instrument.create_group(f'xgm_{i}')
        xgm.attrs['NX_class'] = 'NXcollection'
        xgm['description'] = desc
        for out, key in (('intensity', 'intensityTD'), ('intensity_sigma', 'intensitySigmaTD'),
                         ('x', 'xTD'), ('y', 'yTD'), ('x_sigma', 'xSigmaTD'), ('y_sigma', 'ySigmaTD')):
            copy_meta(xgm, out, meta, ma, f'pulse/{name}_{key}')

    for i, name in enumerate(('SA1_XTD2', 'SPB_XTD9'), 1):
        att = instrument.create_group(f'attenuator_{i}')
        att.attrs['NX_class'] = 'NXattenuator'
        att['type'] = f'{name}_ATT'
        copy_meta(att, 'attenuator_transmission', meta, ma, f'train/attenuator/transmission_{name}')

    es = instrument.create_group('electrospray')
    es.attrs['NX_class'] = 'NXcollection'
    for key in sorted(k for k in meta if k.startswith('train/electrospray/')):
        copy_meta(es, key.split('/')[-1], meta, ma, key)
    for ax in 'xyz':
        copy_meta(es, f'injector_{ax}', meta, ma, f'train/injector/{ax}')
    copy_meta(es, 'chamber_pressure', meta, ma, 'train/chamber_pressure')


def write_detector_static(instrument, geometry, distance, mask):
    corner, basis, xyz, pixel_size, geom_file = geometry
    det = instrument.create_group('detector_1')
    det.attrs['NX_class'] = 'NXdetector'
    det['description'] = 'AGIPD 1M'
    for name, val in (('distance', distance), ('x_pixel_size', pixel_size), ('y_pixel_size', pixel_size)):
        det.create_dataset(name, data=val).attrs['units'] = 'm'
    det.create_dataset('module_identifier',
                       data=[f'AGIPD{m:02d}T{t}' for m in range(NMODULES) for t in range(NTILES)],
                       dtype=h5py.string_dtype()).attrs['description'] = \
        'tile t = rows 64t..64t+63 of AGIPD module mm; per-pixel arrays reshape (C order) to (16, 512, 128)'
    ds = det.create_dataset('corner_position', data=corner)
    ds.attrs.update(units='m', axes='module_identifier:coordinate')
    ds = det.create_dataset('basis_vectors', data=basis)
    ds.attrs.update(units='m', axes='module_identifier:dimension:coordinate')
    ds = det.create_dataset('xyz_map', data=xyz, **GZ)
    ds.attrs.update(units='m', axes='coordinate:module_identifier:y:x',
                    description='pixel centre positions (same geometry as corner_position/basis_vectors)')
    det.create_dataset('quadrant', data=['Q1', 'Q2', 'Q3', 'Q4'], dtype=h5py.string_dtype()).attrs['description'] = \
        'quadrant q is modules 4q..4q+3, i.e. tiles (module_identifier) 32q..32q+31'

    ds = det.create_dataset('mask', data=to_tiles(mask.astype(np.uint32)), **GZ)
    ds.attrs['axes'] = 'module_identifier:y:x'
    ds.attrs['bit_definitions'] = [
        f'0x{CXI_PIXEL_IS_BAD:08x} CXI_PIXEL_IS_BAD: excluded by the analysis good-pixel mask (scratch/det/mask.h5)']
    return det, geom_file


def write_motors_and_note(det, meta, ma, geom_file):
    motors = det.create_group('motors')
    motors.attrs['NX_class'] = 'NXcollection'
    for key in sorted(k for k in meta if k.startswith('train/agipd_motors/')):
        copy_meta(motors, key.split('/')[-1], meta, ma, key)

    note = det.create_group('note_1')
    note.attrs['NX_class'] = 'NXnote'
    note['file_name'] = os.path.basename(REFERENCE_GEOM)
    note['description'] = (
        'Detector geometry: facility reference geometry (this file) moved to the recorded AGIPD quadrant '
        'motor positions with extra_geom.motors.AGIPD_1MMotors, and in z to the recorded detector stage '
        f'position ({os.path.relpath(geom_file, ROOT)}, common.make_geom); identical for all runs in this file.')
    with open(os.path.join(ROOT, REFERENCE_GEOM)) as gf:
        note['data'] = gf.read()


def main():
    p = argparse.ArgumentParser(description='Write a publication CXI file with every hit of one sample')
    p.add_argument('--sample', default='Erythrocruorin')
    p.add_argument('-c', '--conditions', help='per-run-range sample conditions CSV (see merge_cxi.py)')
    p.add_argument('--runs', type=int, nargs='+', help='only these runs of the sample (e.g. for a test)')
    p.add_argument('-o', '--output', help='output file (default scratch/saved_hits/<sample>_publication.cxi)')
    p.add_argument('--description', default='Single particle diffraction patterns of erythrocruorin '
                   'recorded with the AGIPD 1M detector at the SPB/SFX instrument of the European XFEL.')
    args = p.parse_args()

    files = merge_cxi.group_by_sample(merge_cxi.SAVED_HITS_DIR)[args.sample]
    if args.runs:
        files = [(r, path) for r, path in files if r in args.runs]
    runs = np.array([r for r, _ in files])
    output = args.output or os.path.join(merge_cxi.SAVED_HITS_DIR,
                                         f'{merge_cxi.safe_name(args.sample)}_publication.cxi')
    proposal = merge_cxi.get_proposal_from_prefix()
    print(f'{args.sample}: {len(runs)} runs -> {output}')

    # per-event columns, scores, results and fluence, concatenated in run order
    info = merge_cxi.scan_inputs(files)
    N, nevents = info['total'], info['nevents']
    run_col = info['run']
    meta, ma = load_metadata(runs, nevents)

    expid = info['cols']['experiment_identifier'].astype(object)
    expid = np.array([e.decode() if isinstance(e, bytes) else e for e in expid], dtype=object)
    assert len(np.unique(expid)) == N, 'experiment identifiers are not unique'
    timestamps = meta['timestamp']
    if np.any(timestamps == ''):
        sys.exit(f'{np.sum(timestamps == "")} events have no timestamp')

    # every frame's stored pulse energy must be the XGM value at its pulse
    pe_cxi = info['cols']['pulse_energy']
    pe_meta = meta['pulse/XGM_SPB_XTD9_intensityTD']
    if not np.allclose(pe_cxi, pe_meta, rtol=1e-4, equal_nan=True):
        sys.exit('CXI pulse_energy disagrees with the metadata XGM energy (run fix_pulse_energy_cxi.py)')

    # good hits, recomputed uniformly (events without classification/sizing: -1 -> not good)
    is_hit = is_good_hit(info['result_1']['cols']['data'], info['result_2']['cols']['data'],
                         info['score']['hit_score_mask'], **GOOD_HIT)
    if 'is_hit' in info['score']:
        old = info['score']['is_hit'].astype(np.float32)
        known = np.isfinite(old)
        print(f'is_hit: {is_hit.sum()} good hits; differs from the per-run is_hit for '
              f'{np.sum(is_hit[known] != (old[known] > 0))} of {known.sum()} events')

    xyz_ref = info['first_det']['xyz_map'][0]
    geometry = load_geometry(runs[0], xyz_ref)
    mask = info['first_det']['mask'][0]
    distance = float(info['first_det']['distance'][0])

    fluence_attrs = {}
    with h5py.File(files[0][1], 'r') as f:
        fp = 'entry_1/instrument_1/source_1/fluence_sizing'
        if fp in f:
            fluence_attrs = {k: f[fp].attrs[k] for k in ('C_max', 'I_max', 'r0', 'tau') if k in f[fp].attrs}

    cond = None
    if args.conditions:
        text, rows = merge_cxi.load_conditions(args.conditions)
        cols, index = merge_cxi.conditions_per_event(rows, run_col)
        cond = dict(cols=cols, index=index, text=text, path=args.conditions)

    with h5py.File(output, 'w') as g:
        g['cxi_version'] = 160
        entry = g.create_group('entry_1')
        entry.attrs['NX_class'] = 'NXentry'
        entry.create_dataset('experiment_identifier', data=expid, dtype=h5py.string_dtype(), **GZ)
        entry['title'] = f'p{proposal:06d} {args.sample} hits, runs {runs[0]}-{runs[-1]}'
        entry['experiment_description'] = args.description
        entry['program_name'] = 'xfel10662/offline/make_publication_cxi.py'
        entry['start_time'] = min(timestamps)
        entry['end_time'] = max(timestamps)

        sample = entry.create_group('sample_1')
        sample.attrs['NX_class'] = 'NXsample'
        sample['name'] = args.sample

        instrument = entry.create_group('instrument_1')
        instrument.attrs['NX_class'] = 'NXinstrument'
        instrument['name'] = 'SPB'
        write_beam_and_injector(instrument, meta, ma, info['fluence'], fluence_attrs)
        if cond is not None:
            merge_cxi.write_conditions(entry, cond)

        det, geom_file = write_detector_static(instrument, geometry, distance, mask)
        link_expid(det)

        # event identifiers
        per_event(det, 'start_time', timestamps.astype(object), description=TIMESTAMP_DESCRIPTION)
        per_event(det, 'run', run_col.astype(np.uint16))
        per_event(det, 'trainId', info['cols']['trainId'].astype(np.uint64))
        per_event(det, 'cellId', info['cols']['cellId'].astype(np.uint16))
        per_event(det, 'pulseId', meta['pulseId'].astype(np.int64), description='AGIPD pulseId (from the VDS)')
        per_event(det, 'vds_index', info['cols']['vds_index'].astype(np.int64),
                  description='frame index in the per-run VDS file')

        # frames (tiled), per-run backgrounds; powder and photon scores on the way
        dst = det.create_dataset('data', shape=(N,) + PANEL_SHAPE, dtype=np.uint8,
                                 chunks=(1,) + PANEL_SHAPE, **GZ)
        dst.attrs.update(axes='experiment_identifier:module_identifier:y:x', signal=1, units='counts',
                         description='photons per pixel, per-cell masked, not masked with detector_1/mask')
        dw = det.create_dataset('data_white', shape=(len(runs),) + PANEL_SHAPE, dtype=np.float32,
                                chunks=(1,) + PANEL_SHAPE, **GZ)
        dw.attrs.update(axes='background_run:module_identifier:y:x', units='counts',
                        description='per-run beamline background: per-pixel mean of the non-hit frames '
                                    '(add_background_cxi.py)')
        powder = np.zeros(PANEL_SHAPE, dtype=np.uint64)
        photons = np.empty(N, dtype=np.float32)
        lit = np.empty(N, dtype=np.float32)
        bweight = np.empty(N, dtype=np.float32)
        bcounts = np.empty(N, dtype=np.float32)
        bmask_score = np.full(len(runs), np.nan, dtype=np.float32)
        off = 0
        block = 256
        for b, ((run, path), n) in enumerate(zip(tqdm(files, desc='runs'), nevents)):
            with h5py.File(path, 'r') as f:
                src = f[f'{DET}/data']
                for s in range(0, n, block):
                    e = min(s + block, n)
                    frames = to_tiles(src[s:e])
                    dst[off + s:off + e] = frames
                    powder += frames.sum(0, dtype=np.uint64)
                    photons[off + s:off + e] = frames.reshape(e - s, -1).sum(1)
                    lit[off + s:off + e] = np.count_nonzero(frames.reshape(e - s, -1), axis=1)
                W = f[f'{DET}/data_white'][()].astype(np.float32)
                dw[b] = to_tiles(W)
                bweight[off:off + n] = f[f'{DET}/background_weighting'][()]
                bcounts[off:off + n] = bweight[off:off + n] * W.sum(dtype=np.float64)
                p = f'{DET}/score/hit_score_mask_data_white'
                if p in f:
                    bmask_score[b] = f[p][()]
            off += n

        pw = det.create_dataset('powder', data=(powder / N).astype(np.float32), **GZ)
        pw.attrs.update(axes='module_identifier:y:x', units='counts',
                        description=f'mean of data over all {N} events (background included)')
        det.create_dataset('background_run', data=runs.astype(np.uint16)).attrs['description'] = \
            'run of each data_white'
        per_event(det, 'background_index', np.repeat(np.arange(len(runs), dtype=np.uint32), nevents),
                  description='index into data_white for this event')
        per_event(det, 'background_weighting', bweight,
                  description='background of this event = background_weighting * data_white[background_index]; '
                              'per-train scale from the non-hit frames times the normalised XGM pulse energy')

        # scores
        score = det.create_group('score')
        for name, col in info['score'].items():
            if name in SCORES_DROPPED:
                continue
            per_event(score, name, col.astype(np.float32), description=SCORE_DESCRIPTIONS.get(name, ''))
        per_event(score, 'is_hit', is_hit, description=IS_HIT_DESCRIPTION)
        per_event(score, 'photon_counts', photons, description='photons in the frame (per-cell mask applied)')
        per_event(score, 'lit_pixels', lit, description='pixels with at least one photon (per-cell mask applied)')
        per_event(score, 'background_counts', bcounts,
                  description='sum over pixels of background_weighting * data_white[background_index]')
        ds = score.create_dataset('hit_score_mask_data_white', data=bmask_score)
        ds.attrs.update(axes='background_run',
                        description='photons of data_white in the hit-finding mask; the background '
                                    'contribution to hit_score_mask is background_weighting times this')
        for name, key in (('facility_hit_flag', 'pulse/hitfinder_hitFlag'),
                          ('facility_miss_flag', 'pulse/hitfinder_missFlag'),
                          ('facility_threshold_mu', 'train/hitfinder_threshold_mu'),
                          ('facility_threshold_sigma', 'train/hitfinder_threshold_sig'),
                          ('facility_lit_pixels', 'pulse/litpx_litPixels'),
                          ('facility_total_intensity', 'pulse/litpx_totalIntensity'),
                          ('facility_unmasked_pixels', 'pulse/litpx_unmaskedPixels')):
            copy_meta(score, name, meta, ma, key)

        write_motors_and_note(det, meta, ma, geom_file)

        if info['result_1'] is not None:
            merge_cxi.write_result_group(entry, 'result_1', info['result_1'], 'make_publication_cxi.py')
        if info['result_2'] is not None:
            merge_cxi.write_result_group(entry, 'result_2', info['result_2'], 'make_publication_cxi.py')

        data_1 = entry.create_group('data_1')
        data_1.attrs['NX_class'] = 'NXdata'
        data_1['data'] = h5py.SoftLink(f'/{DET}/data')
        link_expid(data_1)

        process = entry.create_group('process_1')
        process.attrs['NX_class'] = 'NXprocess'
        process['program'] = 'make_publication_cxi.py'
        process['version'] = git_version()
        process['date'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        process['command'] = command_line()

    print(f'written {output}: {N} events, {len(runs)} runs')


if __name__ == '__main__':
    main()
