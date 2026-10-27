"""
Collect per-event metadata for a per-run hits CXI file from the raw/proc data,
written to a sidecar next to it for make_publication_cxi.py:

    {PREFIX}scratch/saved_hits/r{run:04d}_hits.cxi.meta.h5

Every per-event array has the CXI file's event axis. Each quantity is taken at
the finest resolution recorded: per pulse where available, otherwise the
per-train value of the event's train. Values are converted to SI where the
facility unit is known; each dataset carries `units`, `source` and `key`
attributes (`units` 'unknown ...' where the facility data records none).

Pulse alignment (checked on r0013 and r0216 against LITFRM energyPerFrame):
    the AGIPD frame with pulseId p is LITFRM frame slot c, detectorPulseId[c] == p
    its XGM pulse slot is xgmPulseId[cumsum(nPulsePerFrame)[c] - 1]
    (frames with nPulsePerFrame == 0 saw no pulse: XGM values NaN)

Layout:
    pulseId                   int64   AGIPD pulseId (from the VDS)
    timestamp                 string  approximate train time, ISO 8601 (+-1 s):
                                      t0 + 0.1 s * trainId, t0 the median over the
                                      run of (recorded train time - 0.1 s * trainId)
    pulse/<name>              per-pulse quantities (XGM, LITFRM, hit finder, lit pixels)
    train/<name>              per-train control values (electrospray, motors, ...)

Usage (on maxwell):
    python get_event_metadata.py 216                 # one run
    python get_event_metadata.py -n 16               # every r####_hits.cxi of the sample
    python get_event_metadata.py --sample Erythrocruorin -n 16
"""
import argparse
import multiprocessing as mp
import traceback

import numpy as np
import h5py
import extra_data
import scipy.constants as sc

from constants import PREFIX
import make_cxi_file as mcf
import merge_cxi

DET_PATH = 'entry_1/instrument_1/detector_1'
CORR_SRC = 'SPB_DET_AGIPD1M-1/CORR/{}CH0:output'
NMODULES = 16
DATA_SELECTOR = 'SPB_IRU_AGIPD1M1/MDL/DATA_SELECTOR'

# per-pulse XGM keys, read at the XGM pulse slot of each frame
XGM_KEYS = ('data.intensityTD', 'data.intensitySigmaTD', 'data.xTD', 'data.yTD',
            'data.xSigmaTD', 'data.ySigmaTD')
XGM_SRCS = {
    'XGM_SPB_XTD9': 'SPB_XTD9_XGM/XGM/DOOCS:output',   # downstream, near the sample
    'XGM_SA1_XTD2': 'SA1_XTD2_XGM/XGM/DOOCS:output',   # upstream
}

# per-train control values, written under train/<name>
CONTROL_SRCS = {
    'electrospray/CO2_capillary': ('SPB_IRU_AEROSOL/FLOW/CO2_CAPILLARY', 'measureCapacity'),
    'electrospray/CO2_chamber':   ('SPB_IRU_AEROSOL/FLOW/CO2_CHAMBER',   'measureCapacity'),
    'electrospray/DP1':           ('SPB_IRU_AEROSOL/FLOW/DP_1',          'measureCapacity'),
    'electrospray/DP2':           ('SPB_IRU_AEROSOL/FLOW/DP_2',          'measureCapacity'),
    'electrospray/He_chamber':    ('SPB_IRU_AEROSOL/FLOW/HE_CHAMBER',    'measureCapacity'),
    'electrospray/N2_capillary':  ('SPB_IRU_AEROSOL/FLOW/N2_CAPILLARY',  'measureCapacity'),
    'electrospray/N2_chamber':    ('SPB_IRU_AEROSOL/FLOW/N2_CHAMBER',    'measureCapacity'),
    'electrospray/liquidjet_He':  ('SPB_IRU_LIQUIDJET/FLOW/HE',          'measureCapacity'),
    'electrospray/voltage':       ('SPB_EXP_HV/MDL/SHQ1',                'channel1.voltage'),
    'electrospray/current':       ('SPB_EXP_HV/MDL/SHQ1',                'channel1.current'),
    'chamber_pressure':           ('SPB_IRU_VAC/GAUGE/GAUGE_FR_6',       'value'),
    'injector/x':                 ('SPB_IRU_INJMOV/MOTOR/X',             'actualPosition'),
    'injector/y':                 ('SPB_IRU_INJMOV/MOTOR/Y',             'actualPosition'),
    'injector/z':                 ('SPB_IRU_INJMOV/MOTOR/Z',             'actualPosition'),
    'attenuator/transmission_SA1_XTD2': ('SA1_XTD2_ATT/MDL/MAIN',        'actual.transmission'),
    'attenuator/transmission_SPB_XTD9': ('SPB_XTD9_ATT/MDL/MAIN',        'actual.transmission'),
    'undulator_energy':           ('SPB_XTD2_UND/DOOCS/ENERGY',          'actualPosition'),
}
for q in range(1, 5):
    for m in range(1, 3):
        CONTROL_SRCS[f'agipd_motors/Q{q}M{m}'] = (DATA_SELECTOR, f'spbIruAgipd1MMotorQ{q}M{m}.actualPosition')
CONTROL_SRCS['agipd_motors/z_stepper'] = (DATA_SELECTOR, 'spbIruAgipd1MMotorZStepper.actualPosition')

# units not recorded in the raw data
MISSING_UNITS = {
    'chamber_pressure': 'unknown (likely mbar)',
    'attenuator/transmission_SA1_XTD2': 'dimensionless',
    'attenuator/transmission_SPB_XTD9': 'dimensionless',
}

# facility unit -> (SI unit, scale)
TO_SI = {
    'μJ': ('J', 1e-6), 'uJ': ('J', 1e-6),
    'mm': ('m', 1e-3),
    'nm': ('m', 1e-9),
    'keV': ('J', 1e3 * sc.e),
}

TRAIN_PERIOD_NS = 100_000_000


def get_units(kd):
    try:
        return str(kd.units or '')
    except Exception:
        return ''


def to_si(vals, units):
    if units in TO_SI:
        si, scale = TO_SI[units]
        return vals * scale, si
    return vals, units


def lookup(src_keys, target_keys):
    """index into src_keys for each target key, and a mask of which were found"""
    src_keys = np.asarray(src_keys, dtype=np.int64)
    order = np.argsort(src_keys)
    pos = np.clip(np.searchsorted(src_keys[order], target_keys), 0, max(len(order) - 1, 0))
    found = (len(order) > 0) & (src_keys[order][pos] == target_keys)
    return order[pos], found


def per_train_rows(kd, tids):
    return lookup(kd.train_id_coordinates(), tids)


class Columns:
    """per-event output columns"""
    def __init__(self, N):
        self.N = N
        self.data = {}
        self.attrs = {}

    def set(self, name, idx, vals, units='', source='', key='', dtype=np.float32, fill=np.nan):
        if name not in self.data:
            self.data[name] = np.full(self.N, fill, dtype=dtype)
        self.data[name][idx] = vals
        self.attrs.setdefault(name, dict(units=units, source=source, key=key))


def approx_timestamps(dc, trainId):
    """ISO 8601 time of each event's train, t0 + 0.1 s * trainId (see module doc)"""
    tids = np.asarray(dc.train_ids, dtype=np.int64)
    ts = np.asarray(dc.train_timestamps())
    known = ~np.isnat(ts)
    if not known.any():
        return np.full(len(trainId), '', dtype=object)
    t0 = np.median(ts[known].astype('datetime64[ns]').astype(np.int64) - tids[known] * TRAIN_PERIOD_NS)
    t = np.int64(np.rint(t0)) + trainId.astype(np.int64) * TRAIN_PERIOD_NS
    return np.array([np.datetime_as_string(np.datetime64(int(x), 'ns'), unit='ms') + 'Z' for x in t],
                    dtype=object)


def process_run(dc, trainId, pulseId, out):
    N = len(trainId)
    ev = np.arange(N)
    t = trainId

    # LITFRM: frame slot of each event, its XGM pulse slot, and LITFRM energies
    lf = dc[mcf.LITFRM_SRC]
    rows, ok = per_train_rows(lf['data.detectorPulseId'], t)
    dpid = lf['data.detectorPulseId'].ndarray().astype(np.int64)
    npf = lf['data.nPulsePerFrame'].ndarray().astype(np.int64)
    xpid = lf['data.xgmPulseId'].ndarray().astype(np.int64)
    rank = np.cumsum(npf, axis=1) - 1

    slot = np.full(N, -1, dtype=np.int64)
    for i in np.where(ok)[0]:
        c = np.nonzero(dpid[rows[i]] == pulseId[i])[0]
        if len(c):
            slot[i] = c[0]
    ok &= slot >= 0
    r, c = rows[ok], slot[ok]

    xgm_slot = np.full(N, -1, dtype=np.int64)
    has = npf[r, c] > 0
    xgm_slot[np.where(ok)[0][has]] = xpid[r[has], rank[r[has], c[has]]]

    for key in ('data.energyPerFrame', 'data.energySigma'):
        kd = lf[key]
        vals, units = to_si(kd.ndarray()[r, c].astype(float), get_units(kd))
        out.set(f'pulse/LITFRM_{key.split(".")[-1]}', ev[ok], vals, units, mcf.LITFRM_SRC, key)
    out.set('pulse/LITFRM_nPulsePerFrame', ev[ok], npf[r, c], '', mcf.LITFRM_SRC,
            'data.nPulsePerFrame', dtype=np.int16, fill=-1)

    # XGMs at the frame's XGM pulse slot
    for name, src in XGM_SRCS.items():
        if src not in dc.all_sources:
            continue
        for key in XGM_KEYS:
            kd = dc[src, key]
            rows_x, okx = per_train_rows(kd, t)
            okx &= xgm_slot >= 0
            vals, units = to_si(kd.ndarray()[rows_x[okx], xgm_slot[okx]].astype(float), get_units(kd))
            out.set(f'pulse/{name}_{key.split(".")[-1]}', ev[okx], vals, units, src, key)

    # facility hit finder, matched on (trainId, pulseId)
    if mcf.HITFINDER_SRC in dc.all_sources:
        hf = dc[mcf.HITFINDER_SRC]
        h_t, h_p = mcf._pulse_resolved(hf['data.pulseId'])
        okh, idx = mcf._align_by_pair(t, pulseId, h_t, h_p)
        for key, dtype, fill in (('data.hitscore', np.float32, np.nan),
                                 ('data.hitFlag', np.int8, -1), ('data.missFlag', np.int8, -1)):
            _, vals = mcf._pulse_resolved(hf[key])
            out.set(f'pulse/hitfinder_{key.split(".")[-1]}', ev[okh], vals[idx[okh]],
                    '', mcf.HITFINDER_SRC, key, dtype=dtype, fill=fill)
        for key in ('threshold.mu', 'threshold.sig'):
            rows_h, okt = per_train_rows(hf[key], t)
            out.set(f'train/hitfinder_{key.replace(".", "_")}', ev[okt],
                    hf[key].ndarray().ravel()[rows_h[okt]], '', mcf.HITFINDER_SRC, key)

    # lit pixels / total intensity from the correction pipeline, summed over modules
    sums = {k: np.zeros(N) for k in ('litPixels', 'totalIntensity', 'unmaskedPixels')}
    nmod = np.zeros(N, dtype=np.int8)
    for m in range(NMODULES):
        src = CORR_SRC.format(m)
        if src not in dc.all_sources or 'litpx.litPixels' not in dc[src].keys():
            continue
        s = dc[src]
        idx, okm = lookup(mcf._pair_key(s['litpx.trainId'].ndarray(), s['litpx.pulseId'].ndarray()),
                          mcf._pair_key(t, pulseId))
        for k in sums:
            sums[k][okm] += s[f'litpx.{k}'].ndarray()[idx[okm]]
        nmod += okm
    okl = nmod > 0
    for k, v in sums.items():
        out.set(f'pulse/litpx_{k}', ev[okl], v[okl], '', CORR_SRC.format('*'), f'litpx.{k}')
    out.set('pulse/litpx_nmodules', ev, nmod, '', CORR_SRC.format('*'), 'modules summed',
            dtype=np.int8, fill=0)

    # per-train control values
    for name, (src, key) in CONTROL_SRCS.items():
        if src not in dc.all_sources or key not in dc[src].keys(inc_timestamps=False):
            print(f'warning: {src} {key} not in run; train/{name} skipped')
            continue
        kd = dc[src, key]
        rows_c, okc = per_train_rows(kd, t)
        units = get_units(kd) or MISSING_UNITS.get(name, 'unknown')
        vals, units = to_si(kd.ndarray().ravel()[rows_c[okc]].astype(float), units)
        out.set(f'train/{name}', ev[okc], vals, units, src, key)


def write_run(run, proposal):
    cxi_file = f'{PREFIX}scratch/saved_hits/r{run:04d}_hits.cxi'
    vds_file = f'{PREFIX}scratch/vds/r{run:04d}.cxi'
    out_file = f'{cxi_file}.meta.h5'

    with h5py.File(cxi_file, 'r') as f:
        trainId = f[f'{DET_PATH}/trainId'][()].astype(np.int64)
        vds_index = f[f'{DET_PATH}/vds_index'][()].astype(np.int64)
    with h5py.File(vds_file, 'r') as g:
        assert np.array_equal(g['/entry_1/trainId'][()][vds_index].astype(np.int64), trainId), \
            f'run {run}: VDS trainId != CXI trainId at vds_index'
        pulseId = g['/entry_1/pulseId'][()][vds_index].astype(np.int64)

    dc = extra_data.open_run(proposal, run, data='all')
    timestamps = approx_timestamps(dc, trainId)
    dc = dc.select_trains(extra_data.by_id[np.unique(trainId)])

    out = Columns(len(trainId))
    process_run(dc, trainId, pulseId, out)

    with h5py.File(out_file, 'w') as f:
        f.attrs['input'] = cxi_file
        f.attrs['run'] = run
        f['pulseId'] = pulseId
        f.create_dataset('timestamp', data=list(timestamps.astype(str)), dtype=h5py.string_dtype())
        for name, v in out.data.items():
            ds = f.create_dataset(name, data=v, compression='gzip')
            for k, a in out.attrs[name].items():
                ds.attrs[k] = a

    # the XGM energy at the frame's pulse must reproduce LITFRM's own value
    e_xgm = out.data.get('pulse/XGM_SPB_XTD9_intensityTD')
    e_lf = out.data.get('pulse/LITFRM_energyPerFrame')
    agree = np.nan
    if e_xgm is not None and e_lf is not None:
        m = np.isfinite(e_xgm) & np.isfinite(e_lf)
        agree = np.mean(np.isclose(e_xgm[m], e_lf[m], rtol=1e-4)) if m.any() else np.nan
    return f'r{run:04d}: {len(trainId)} events, XGM == LITFRM energy for {agree:.3f} of them -> {out_file}'


def _worker(task):
    run, proposal = task
    try:
        return write_run(run, proposal)
    except Exception:
        return f'r{run:04d} ERROR:\n{traceback.format_exc()}'


def main():
    p = argparse.ArgumentParser(description='Write per-event metadata sidecars for per-run hit CXI files')
    p.add_argument('run', type=int, nargs='*', help='run numbers (default: every run of --sample)')
    p.add_argument('--sample', default='Erythrocruorin', help='sample name (default Erythrocruorin)')
    p.add_argument('-n', '--nproc', type=int, default=1, help='runs processed in parallel')
    args = p.parse_args()

    runs = args.run or [r for r, _ in merge_cxi.group_by_sample(merge_cxi.SAVED_HITS_DIR)[args.sample]]
    proposal = mcf.get_proposal_from_prefix()
    with mp.Pool(args.nproc) as pool:
        for msg in pool.imap(_worker, [(r, proposal) for r in runs]):
            print(msg, flush=True)


if __name__ == '__main__':
    main()
