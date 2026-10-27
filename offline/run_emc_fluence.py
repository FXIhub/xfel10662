"""
Re-fit a background-aware per-frame fluence weight w_d for an Ery sizing run and
convert it into an incident beam fluence (photons/m^2) per event.

Why this is a separate step
---------------------------
Sizing (run_emc_sizing.py / config_sizing.py) uses likelihood='fluence_free',
which marginalises the per-frame fluence w_d out of the likelihood, so the w_d it
leaves behind is not a usable fluence estimate.

This runs *in the sizing working directory* (scratch/sizing/r####/, set up by
submit_sizing.sh) and reuses everything sizing already computed -- the same
config.pickle (so get_frames(), the Ery_075nm.h5 model and the scale grid are
identical) and the existing logR matrix (class_logR_0.h5). The only extra work is:

  1. renormalise the existing logR -> P_dr with P_thresh_frames = 1, which keeps
     only the single most-likely term per frame (probability.keep_n_frames) -- a
     one-hot assignment at sizing's most-likely orientation+scale; and
  2. refine w_d with emc3's background Poisson solver
     (emc3.background.update_w_d.w_update), which fits

        F_dri = w_d C_i W_ri + B_di      B_di = background_weighting[d] * data_white

     for w_d on that frozen most-likely tomogram (full measured q-range).

(If the logR matrix / config.pickle are missing, they are rebuilt -- so the script
also works standalone, just more slowly.)

w_d -> fluence
--------------
emc3's frame model uses C_i = (Omega*P)/C_max (emc3.detector) and the sizing model
W_ri = |F(q)|^2 / I_max (class_model_0.h5 == Ery_075nm.h5). Matching the Thomson
per-pixel count n = tau*I0*P*Omega*r0^2*|F|^2 term-by-term gives

    I0 [photons/m^2] = w_d / (tau * r0^2 * C_max * I_max)        (tau = 1)

  * C_max = max(Omega*P) over the run geometry (Detector_cxi, scale_correction off)
  * I_max = the normalisation applied when Ery_075nm.h5 was built from the |F|^2
    pickle, recovered as median(|F|^2 / model.data). Gas-phase compaction is mass
    conserving (F'(q)=F(a q)), so the forward peak |F(0)|^2 = N_e^2 is invariant:
    one I_max serves every frame and there is no volume/Jacobian factor.

Per-event w_d and fluence (photons/m^2, -1 where a frame was not sized) are written
back into the CXI (run.cxi -> r####_hits.cxi):
    entry_1/result_2/w_d
    entry_1/instrument_1/source_1/fluence_sizing

Run like run_emc_sizing.py: `mpirun python run_emc_fluence.py` from the working
directory set up by submit_fluence.sh.
"""
import pickle
import runpy
import sys
import datetime
from pathlib import Path
from time import time

import h5py
import numpy as np

import emc3
from emc3.run import calculate_logR
from emc3.probability import calculate_P
from emc3.background.update_w_d import w_update

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

config_file   = 'config.pickle'
config_script = 'config.py'
cxi_file      = 'run.cxi'
model_file    = 'class_model_0.h5'         # = Ery_075nm.h5 ; data = |F|^2 / I_max
mask_file     = 'mask.h5'
pickle_file   = 'model_intensity.pickle'   # -> Ery_075_aligned_..._N_192.pickle

# Hardening temperature for the assignment. With P_thresh_frames=1 only the top
# term per frame is kept, but a sharp beta also makes the renormalised P_dr a true
# one-hot (exp arg <= 0, so large beta is numerically safe).
beta = 1.0e3

# Fluence calibration constants.
tau = 1.0                       # detector quantum efficiency (AGIPD ~ fully absorbs 6 keV)
r0  = 2.8179403262e-15          # classical electron radius [m]


def recover_I_max(model_file, pickle_file):
    """I_max = the normalisation applied when the sizing model (|F|^2 / I_max) was
    made, recovered from the |F|^2 pickle as the (constant) ratio |F|^2/model.data
    over the high-signal voxels. Returns (I_max, I_dc, spread):

      I_dc   = |F(0)|^2 (centre voxel) = N_e^2, the compaction-invariant forward
               peak; equals I_max when the model was normalised by its max (q=0).
      spread = std/median of the ratio (should be ~0 if model == |F|^2/I_max).
    """
    with h5py.File(model_file) as f:
        model = f['data'][()].astype(np.float64)
    with open(pickle_file, 'rb') as fh:
        t = pickle.load(fh)
    F = np.asarray(t['electron_density_fourier'])
    I_full = np.abs(F) ** 2
    if I_full.shape != model.shape:
        raise ValueError(f'pickle intensity {I_full.shape} != model {model.shape}')
    m = model > 0.1 * model.max()
    ratio = I_full[m] / model[m]
    I_max = float(np.median(ratio))
    spread = float(np.std(ratio) / np.median(ratio))
    c = model.shape[0] // 2
    I_dc = float(I_full[c, c, c])
    return I_max, I_dc, spread


# ---- reuse the sizing dir; rebuild config + logR only if missing -------------
# remove any stale fluence chunks from an interrupted run (everything else --
# config.pickle, data_*.h5, class_logR_0.h5 -- is reused).
if rank == 0:
    for fnam in Path('.').glob('fluence_chunk_*.h5'):
        fnam.unlink(missing_ok=True)
comm.Barrier()

# decide once (on rank 0, broadcast) whether config.pickle must be rebuilt, so
# the logR-reuse decision is consistent across ranks and not racy.
rebuild = (not Path(config_file).exists()) if rank == 0 else None
rebuild = comm.bcast(rebuild, root=0)
if rank == 0 and rebuild:
    runpy.run_path(config_script)['make_pickle_file'](output_file=config_file)
comm.Barrier()

config = pickle.load(open(config_file, 'rb'))
config['P_thresh_frames'] = 1      # keep only the most-likely term per frame
config['update_fluence'] = True

# initialise per-frame fluence weights (overwritten by w_update below).
# Also seed b_d from the CXI's *current* background_weighting: the sizing config
# froze b_d into its back_*.h5 files when it was built, and emc3's
# BackCXI.load_from_file() takes b_d from ./fluence.h5 in preference to those, so
# a corrected background_weighting (fix_pulse_energy_cxi.py) reaches the w_d fit
# without rebuilding config.pickle and the logR matrix.
if rank == 0:
    frames = np.asarray(config['classes'][0]['data'].frames)
    with h5py.File(cxi_file, 'r') as f:
        b_d = f['entry_1/instrument_1/detector_1/background_weighting'][()][frames]
    B_di = config['classes'][0]['data'].B_di
    b_d = B_di._floor_positive(b_d.astype(np.float32), B_di.floor_rel)
    if Path(config['fluence_file']).resolve() != Path('fluence.h5').resolve():
        raise RuntimeError(f"fluence_file {config['fluence_file']} is not ./fluence.h5, "
                           'so emc3 would not pick up the refreshed b_d')
    with h5py.File(config['fluence_file'], 'w') as f:
        f['w_d'] = np.ones(len(frames), dtype=float)
        f['b_d'] = b_d
comm.Barrier()

# ---- logR: reuse sizing's matrix if present, else recompute ------------------
logR_present = (not rebuild) and all(
    Path(c['logR_file']).exists() for c in config['classes'])
if not logR_present:
    cids = list(range(len(config['classes'])))
    my_classes = cids[rank::size]
    t0 = time()
    calculate_logR(config_file, config, p_per_device=2, cids=my_classes)
    comm.Barrier()
    if rank == 0:
        print(f'logR (recomputed): {time() - t0:.1f}s', flush=True)
elif rank == 0:
    print('reusing existing logR matrix', flush=True)

# ---- renormalise logR -> one-hot P_dr (cheap) -------------------------------
if rank == 0:
    calculate_P(config, beta)
comm.Barrier()

# ---- background-aware Poisson w_d on the frozen most-likely tomogram ----------
t0 = time()
w_update(config, config_file)
comm.Barrier()
if rank == 0:
    print(f'w_d: {time() - t0:.1f}s', flush=True)

# ---- calibrate and write the per-event fluence back into the CXI (rank 0) -----
if rank == 0:
    with h5py.File(config['fluence_file']) as f:
        w_d = f['w_d'][()]                       # (D,) compact frame order

    frames_d = np.asarray(config['classes'][0]['data'].frames)   # CXI event index

    # C_max and I_max for the calibration; keep the (expensive) w_d even if these
    # fail so the run is not wasted -- fluence can be recomputed from result_2/w_d.
    fluence = None
    C_max = I_max = None
    try:
        with h5py.File(mask_file) as f:
            mask = f['data'][()]
        det = emc3.Detector_cxi(cxi_file, mask, scale_correction=False)
        C_max = float(det.C.max())               # max(Omega * P)

        I_max, I_dc, spread = recover_I_max(model_file, pickle_file)
        print(f'C_max={C_max:.6e}  I_max={I_max:.6e}  |F(0)|^2={I_dc:.6e} '
              f'(N_e={I_dc ** 0.5:.3e})  ratio spread={spread:.2%}', flush=True)

        fluence = w_d / (tau * r0 ** 2 * C_max * I_max)     # photons / m^2
    except Exception as e:
        print(f'WARNING: could not compute fluence ({e}); writing w_d only',
              flush=True)

    with h5py.File(cxi_file, 'r+') as f:
        entry = f['entry_1']
        n_events = entry['instrument_1/detector_1/data'].shape[0]

        if 'result_2' in entry:
            result = entry['result_2']
        else:
            result = entry.create_group('result_2')
            result.create_dataset('frames', data=frames_d.astype(np.int64))

        w_event = np.full(n_events, -1.0, dtype=float)
        w_event[frames_d] = w_d
        if 'w_d' in result:
            del result['w_d']
        result.create_dataset('w_d', data=w_event)

        if fluence is not None:
            flu_event = np.full(n_events, -1.0, dtype=float)
            flu_event[frames_d] = fluence
            src = entry['instrument_1/source_1']
            if 'fluence_sizing' in src:
                del src['fluence_sizing']
            ds = src.create_dataset('fluence_sizing', data=flu_event)
            ds.attrs['units'] = 'photons/m^2'
            ds.attrs['C_max'] = C_max
            ds.attrs['I_max'] = I_max
            ds.attrs['tau'] = tau
            ds.attrs['r0'] = r0

        if 'fluence_process' in result:
            del result['fluence_process']
        proc = result.create_group('fluence_process')
        proc['program'] = 'run_emc_fluence.py'
        proc['command'] = ' '.join(sys.argv)
        proc['date'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        proc['version'] = getattr(emc3, '__version__', 'unknown')
        proc['note'] = ('background B_di = background_weighting * data_white, with '
                        'background_weighting read from this CXI at fit time')

    if fluence is not None:
        fmed = float(np.median(fluence)) if len(fluence) else float('nan')
        print(f'wrote fluence for {len(frames_d)} events; '
              f'median {fmed:.3e} ph/m^2 ({fmed * 1e-12:.3e} ph/um^2)', flush=True)
    else:
        print(f'wrote w_d for {len(frames_d)} events (fluence skipped)', flush=True)
