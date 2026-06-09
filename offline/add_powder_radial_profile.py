"""
Radial profile of the good-hit powder vs momentum transfer q, compared with the
radial profile of a 3D reference intensity model.

The powder (per-pixel sum of the good-hit frames, written by add_is_hit_cxi.py)
is azimuthally averaged over the good pixels, binned in q. q is computed with the
same convention emc3 uses for its detector/model grid (see emc3.detector):

    q_vec = (xyz / |xyz| - z_hat) / lambda      |q| = 2 sin(theta) / lambda   [1/m]

so the resolution is d = 1 / |q|. The reference model lives on a cubic q-grid
({PREFIX}scratch/models/Ery_075nm.h5: /data is (N,N,N), /dq is the q side-length
of one voxel in 1/m), centred at q = 0 (index N//2). Its profile is the spherical
average of /data, with voxel radius k mapping to q = dq * k. Because the model
was reconstructed from different geometry, its dq differs from any single run's
detector dq, so the two profiles are placed on their own physical q-axes (1/m)
and overlaid vs physical q rather than shared bins.

The beamline background is subtracted: each good-hit frame d carries a background
of background_weighting[d] * data_white, so the powder contains
(sum_d background_weighting[d]) * data_white over the good hits. The radial
profile of that term is subtracted from the powder's radial profile to isolate
the sample signal (skipped if data_white / background_weighting / is_hit absent).

Written into the CXI (overwriting, so reprocessing is idempotent):
    /entry_1/instrument_1/detector_1/powder_radial_profile             (bg-subtracted)
    /entry_1/instrument_1/detector_1/powder_radial_profile_q           (q, 1/m)
    /entry_1/instrument_1/detector_1/powder_radial_profile_raw         (before subtraction)
    /entry_1/instrument_1/detector_1/powder_radial_profile_background  (subtracted term)

Usage
-----
    python add_powder_radial_profile.py 12                    # r0012_hits.cxi
    python add_powder_radial_profile.py 12 --png prof.png     # also save a plot
    python add_powder_radial_profile.py -c some.cxi -m model.h5

NB: no module-level imports of the offline-local `constants` module, nor of
matplotlib. This file is also exec'd as a library by DAMNIT's context.py
(powder_radial_profile), where the offline/ dir is not on sys.path; the CLI needs
constants only inside parse_args(), and the plot only with --png.
"""

import argparse

import h5py
import numpy as np
import scipy.constants as sc

DET_PATH        = 'entry_1/instrument_1/detector_1'
PROFILE_PATH    = f'{DET_PATH}/powder_radial_profile'             # background-subtracted
Q_PATH          = f'{DET_PATH}/powder_radial_profile_q'           # q, 1/m
RAW_PATH        = f'{DET_PATH}/powder_radial_profile_raw'         # before subtraction
BACKGROUND_PATH = f'{DET_PATH}/powder_radial_profile_background'  # subtracted term


def _wavelength(f):
    """Photon wavelength in metres, matching emc3.detector.Detector_cxi:
    legacy photon_wavelength (m) takes precedence, else derive from energy (J)."""
    k_wl = 'entry_1/instrument_1/source_1/photon_wavelength'
    k_en = 'entry_1/instrument_1/source_1/energy'
    if k_wl in f:
        return float(f[k_wl][0])
    if k_en in f:
        return float(sc.h * sc.c / f[k_en][0])
    raise KeyError(f'CXI has neither {k_wl} nor {k_en}')


def pixel_q(xyz, wavelength):
    """Per-pixel |q| (1/m), q = (xyz/|xyz| - z_hat)/lambda  (emc3 convention)."""
    r = np.sqrt(np.sum(xyz**2, axis=0))
    q = xyz / r
    q[2] -= 1.0
    q /= wavelength
    return np.sqrt(np.sum(q**2, axis=0))


def radial_average(values, qr, dq, good):
    """Azimuthal average of `values` over `good` pixels, binned at q-step `dq`.

    Returns (q, profile) with q = dq * arange(nbins) in the units of `qr`/`dq`."""
    k = np.rint(qr / dq).astype(int)
    sel = good & np.isfinite(values)
    counts = np.bincount(k[sel], minlength=1)
    total  = np.bincount(k[sel], weights=values[sel], minlength=len(counts))
    profile = np.where(counts > 0, total / np.clip(counts, 1, None), np.nan)
    q = dq * np.arange(len(profile))
    return q, profile


def compute_powder_profile(cxi_file):
    """Radial profile of the good-hit powder, background-subtracted.

    Each good-hit frame d carries a beamline background of
    background_weighting[d] * data_white, so the total background in the powder is
    (sum_d background_weighting[d]) * data_white over the good hits. We subtract
    the radial profile of that term from the powder's radial profile to isolate
    the sample signal. If data_white / background_weighting / is_hit are absent,
    no subtraction is done and `background` is None.

    Returns (q [1/m], profile, raw, background): `profile` is raw - background."""
    with h5py.File(cxi_file) as f:
        if f'{DET_PATH}/powder' not in f:
            raise RuntimeError(
                f'{cxi_file} has no {DET_PATH}/powder; run add_is_hit_cxi.py first')
        powder = f[f'{DET_PATH}/powder'][()].astype(np.float64)
        xyz    = f[f'{DET_PATH}/xyz_map'][()]
        good   = f[f'{DET_PATH}/mask'][()] == 0
        wav    = _wavelength(f)
        z      = float(xyz[2].ravel()[0])
        pixel_size = float(f[f'{DET_PATH}/x_pixel_size'][()])

        # background pieces (all three needed to subtract)
        data_white = bw = is_hit = None
        if f'{DET_PATH}/data_white' in f:
            data_white = f[f'{DET_PATH}/data_white'][()].astype(np.float64)
        if f'{DET_PATH}/background_weighting' in f:
            bw = f[f'{DET_PATH}/background_weighting'][()].astype(np.float64)
        if f'{DET_PATH}/score/is_hit' in f:
            is_hit = f[f'{DET_PATH}/score/is_hit'][()].astype(bool)

    qr = pixel_q(xyz, wav)
    # bin at the detector's low-angle q-step, matching emc3's voxel size for this
    # geometry (emc3.detector.Detector.dq = pixel_size / lambda / z).
    dq = pixel_size / wav / z
    q, raw = radial_average(powder, qr, dq, good)

    background = None
    if data_white is not None and bw is not None and is_hit is not None:
        # total background weight summed over the good hits that built the powder
        w_sum = float(np.nansum(bw[is_hit]))
        _, white = radial_average(data_white, qr, dq, good)
        background = w_sum * white

    profile = raw if background is None else raw - background
    return q, profile, raw, background


def compute_model_profile(model_file, qmax=None):
    """Spherical average of the 3D model intensities vs q (1/m).

    /data is (N,N,N) centred at q=0 (index N//2); /dq is the voxel q side-length.
    Optionally truncate to q <= qmax."""
    with h5py.File(model_file) as f:
        data = f['data'][()].astype(np.float64)
        dq   = float(f['dq'][()])

    n = data.shape[0]
    c = n // 2
    ax = np.arange(n) - c
    # integer voxel radius; bincount over it gives the spherical shell average
    k = np.rint(np.sqrt(ax[:, None, None]**2 + ax[None, :, None]**2
                        + ax[None, None, :]**2)).astype(int)
    counts = np.bincount(k.ravel(), minlength=1)
    total  = np.bincount(k.ravel(), weights=data.ravel(), minlength=len(counts))
    profile = total / np.clip(counts, 1, None)
    q = dq * np.arange(len(profile))

    if qmax is not None:
        keep = q <= qmax
        q, profile = q[keep], profile[keep]
    return q, profile


def add_powder_radial_profile(cxi_file, model_file=None):
    """Compute the background-subtracted powder radial profile, write it (plus the
    raw profile and the subtracted background term) into the CXI, and (if a model
    file is given) also return the model's radial profile on the same physical
    q-axis (1/m), truncated to the powder's q-range.

    Returns (q_powder, profile, raw, background, q_model, model_profile); the
    background is None when no subtraction was possible and the model arrays are
    None when no model file is given."""
    q_p, P, raw, background = compute_powder_profile(cxi_file)

    with h5py.File(cxi_file, 'r+') as f:
        writes = [(PROFILE_PATH, P), (Q_PATH, q_p), (RAW_PATH, raw)]
        if background is not None:
            writes.append((BACKGROUND_PATH, background))
        for path, data in writes:
            if path in f:
                del f[path]
            f[path] = data

    q_m = M = None
    if model_file is not None:
        q_m, M = compute_model_profile(model_file, qmax=float(q_p.max()))
    return q_p, P, raw, background, q_m, M


def _scale_model_to_powder(q_p, P, q_m, M):
    """Single least-squares factor putting the model on the powder's scale, fit
    over the overlapping q-range where both profiles are positive."""
    M_on_p = np.interp(q_p, q_m, M, left=np.nan, right=np.nan)
    sel = np.isfinite(M_on_p) & (M_on_p > 0) & (P > 0)
    if not sel.any():
        return 1.0
    return float(np.sum(P[sel] * M_on_p[sel]) / np.sum(M_on_p[sel] ** 2))


def make_powder_profile_figure(q_p, P, raw=None, background=None,
                               q_m=None, M=None, run='?'):
    """Overlay the (background-subtracted) powder and the scaled model radial
    profiles vs q, with a top resolution axis. When the raw profile and the
    subtracted background term are given, they are shown faintly for context.
    q is plotted in 1/nm, resolution in nm. Returns a Figure."""
    import matplotlib.pyplot as plt

    q_p_nm = q_p * 1e-9                    # 1/m -> 1/nm
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 4)
    fig.set_tight_layout(True)

    label = 'powder (good hits, bg-subtracted)' if background is not None \
            else 'powder (good hits)'
    if background is not None and raw is not None:
        ax.plot(q_p_nm, raw, color='0.7', lw=1, label='powder (raw)')
        ax.plot(q_p_nm, background, color='0.7', lw=1, ls='--',
                label='background (data_white x sum bg-weight)')
    ax.plot(q_p_nm, P, label=label)
    if q_m is not None:
        scale = _scale_model_to_powder(q_p, P, q_m, M)
        ax.plot(q_m * 1e-9, M * scale, label='model (scaled)')

    ax.set_yscale('log')
    ax.set_xlabel('q (1/nm)')
    ax.set_ylabel('azimuthal average (counts)')
    ax.set_title(f'Powder vs model radial profile, run {run}')
    ax.legend()
    ax.spines[['right', 'top']].set_visible(False)

    # secondary resolution axis d = 1/q (q in 1/nm -> d in nm). d blows up near
    # q=0, so place ticks at fixed d values that fall inside the plotted q-range.
    qhi = q_p_nm.max()
    qlo = max(q_p_nm[q_p_nm > 0].min(), 1e-6)
    eps = qlo / 100.0                       # keep 1/q finite at the axis margins
    secax = ax.secondary_xaxis(
        'top', functions=(lambda q: 1.0 / np.clip(q, eps, None),
                          lambda d: 1.0 / np.clip(d, eps, None)))
    # floor the tick q so the large-d ticks (tiny q) don't pile up at the left
    q_tick_lo = max(qlo, 0.08)
    d_ticks = [d for d in (10, 5, 3, 2, 1.5, 1, 0.8)
               if q_tick_lo <= 1.0 / d <= qhi]
    secax.set_xticks(d_ticks)
    secax.set_xlabel('resolution d (nm)')
    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description='Radial profile of the good-hit powder vs q, compared with a '
                    '3D reference intensity model, written into the CXI file.')
    p.add_argument('run', type=int, nargs='?',
                   help='run number -> r{run:04d}_hits.cxi in scratch/saved_hits')
    p.add_argument('-c', '--cxi', help='explicit CXI file path (overrides run)')
    p.add_argument('-m', '--model', help='3D model h5 (default scratch/models/Ery_075nm.h5)')
    p.add_argument('--png', help='also save an overlay plot to this path')
    args = p.parse_args()

    if args.cxi:
        args.cxi_file = args.cxi
    elif args.run is not None:
        from constants import PREFIX
        args.cxi_file = f'{PREFIX}scratch/saved_hits/r{args.run:04d}_hits.cxi'
    else:
        p.error('give a run number or --cxi')

    if args.model is None:
        from constants import PREFIX
        args.model = f'{PREFIX}scratch/models/Ery_075nm.h5'
    return args


if __name__ == '__main__':
    args = parse_args()

    q_p, P, raw, background, q_m, M = add_powder_radial_profile(
        args.cxi_file, args.model)
    print(f'wrote {PROFILE_PATH} ({P.shape[0]} q bins) to {args.cxi_file}; '
          f'q_max = {q_p.max():.4g} 1/m (d_min = {1/q_p.max()*1e9:.3g} nm); '
          f'background {"subtracted" if background is not None else "unavailable"}')

    if args.png:
        run = args.run if args.run is not None else '?'
        fig = make_powder_profile_figure(q_p, P, raw, background, q_m, M, run)
        fig.savefig(args.png)
        print(f'saved plot to {args.png}')
