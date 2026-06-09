"""
Show the integrated powder radial profile of the good hits, summed over several
runs (the same quantity DAMNIT's "Powder radial profile" Variable shows per run,
but accumulated across runs into a single profile).

Each run's CXI stores `detector_1/powder`, the per-pixel sum of that run's
good-hit frames (written by add_is_hit_cxi.py / DAMNIT's good_hits Variable), and
carries a beamline background of (sum_d background_weighting[d]) * data_white over
its good hits. To integrate over runs we accumulate, per pixel, the powders and
the background terms across all the requested runs, then azimuthally average each
once vs momentum transfer q (see add_powder_radial_profile.py for the q
convention and the data_white subtraction). The result is one integrated powder
profile, one integrated background profile, and their difference, overlaid with
the scaled 3D reference model.

This script only reads the CXI files; it does not write anything back.

Runs can be given as run numbers, by sample name (--sample, which selects every
hit CXI whose entry_1/sample_1/name matches, using the same grouping as
merge_cxi.py), or as explicit --cxi paths; the selections are combined and
de-duplicated.

Usage
-----
    python show_integrated_powder_profile.py 12 14 17       # integrate r0012/r0014/r0017
    python show_integrated_powder_profile.py --sample "Erythrocruorin"   # all runs of a sample
    python show_integrated_powder_profile.py --list-samples              # list samples + runs
    python show_integrated_powder_profile.py 12 14 --png powder.png   # save instead of show
    python show_integrated_powder_profile.py 12 14 --model none       # no model overlay
    python show_integrated_powder_profile.py -c a.cxi -c b.cxi        # explicit CXI files
"""

import argparse

import h5py
import numpy as np

# This file lives in offline/ alongside add_powder_radial_profile.py, so when run
# as a script (offline/ on sys.path[0]) a plain import works and reuses the
# q-binning, model profile and figure helpers.
import add_powder_radial_profile as prp

DET_PATH = prp.DET_PATH


def integrate_powder_profile(cxi_files):
    """Sum the good-hit powders and beamline-background terms over `cxi_files`,
    then azimuthally average each once.

    The runs are assumed to share one detector geometry (same pixel grid); the
    powders and the per-pixel background term (sum_d background_weighting[d] over
    the good hits, times data_white) are accumulated across runs in pixel space,
    using the AND of the per-run good masks. Wavelength / sample distance from the
    first run set the q-axis; a warning is printed if a later run differs.

    A run is skipped with a warning (rather than aborting) when its CXI cannot be
    opened, has no detector_1/powder yet (add_is_hit_cxi.py not run), or has a
    powder shape that disagrees with the first run's detector geometry.

    Returns (q [1/m], profile, raw, background, n_used): `profile` is raw -
    background, or raw when no used run supplied the data_white background pieces
    (background None). Raises RuntimeError if no run was usable."""
    total_powder = total_bg = good = None
    geom = None            # (xyz, wav, z, pixel_size) from the first run
    any_bg = False
    n_used = 0

    for cxi in cxi_files:
        try:
            f = h5py.File(cxi, 'r')
        except OSError as e:
            print(f'WARNING: skipping {cxi}: cannot open ({e})')
            continue
        with f:
            if f'{DET_PATH}/powder' not in f:
                print(f'WARNING: skipping {cxi}: no {DET_PATH}/powder yet '
                      '(run add_is_hit_cxi.py first)')
                continue
            powder = f[f'{DET_PATH}/powder'][()].astype(np.float64)
            xyz    = f[f'{DET_PATH}/xyz_map'][()]
            g      = f[f'{DET_PATH}/mask'][()] == 0
            wav    = prp._wavelength(f)
            z      = float(xyz[2].ravel()[0])
            pixel_size = float(f[f'{DET_PATH}/x_pixel_size'][()])

            data_white = bw = is_hit = None
            if f'{DET_PATH}/data_white' in f:
                data_white = f[f'{DET_PATH}/data_white'][()].astype(np.float64)
            if f'{DET_PATH}/background_weighting' in f:
                bw = f[f'{DET_PATH}/background_weighting'][()].astype(np.float64)
            if f'{DET_PATH}/score/is_hit' in f:
                is_hit = f[f'{DET_PATH}/score/is_hit'][()].astype(bool)

        if geom is None:
            geom = (xyz, wav, z, pixel_size)
            total_powder = np.zeros_like(powder)
            total_bg     = np.zeros_like(powder)
            good         = g
        else:
            if powder.shape != total_powder.shape:
                print(f'WARNING: skipping {cxi}: powder shape {powder.shape} != '
                      f'{total_powder.shape} (different detector geometry)')
                continue
            _, wav0, z0, _ = geom
            if abs(wav - wav0) > 1e-2 * wav0 or abs(z - z0) > 1e-2 * z0:
                print(f'WARNING: {cxi} wavelength/distance differ from the first '
                      f'run (lambda {wav:.4g} vs {wav0:.4g} m, z {z:.4g} vs '
                      f'{z0:.4g} m); q-axis taken from the first run.')
            good = good & g

        total_powder += powder
        if data_white is not None and bw is not None and is_hit is not None:
            total_bg += float(np.nansum(bw[is_hit])) * data_white
            any_bg = True
        n_used += 1

    if geom is None:
        raise RuntimeError(
            'no usable runs: every CXI was missing, unreadable, or had no powder')

    xyz, wav, z, pixel_size = geom
    qr = prp.pixel_q(xyz, wav)
    dq = pixel_size / wav / z
    q, raw = prp.radial_average(total_powder, qr, dq, good)

    background = None
    if any_bg:
        _, background = prp.radial_average(total_bg, qr, dq, good)
    profile = raw if background is None else raw - background
    return q, profile, raw, background, n_used


def sb_crossing_q(q, profile, background, ratio=0.6):
    """First q (1/m, scanning from low q) where the signal-to-background ratio
    profile/background falls through `ratio`, linearly interpolated between the
    bracketing bins.

    `profile` is the background-subtracted powder (the signal); the ratio starts
    high at low q and the first downward crossing marks the resolution limit.
    Returns None when there is no background, no valid bins, or no such crossing
    (ratio already below `ratio`, or never below it)."""
    if background is None:
        return None
    q = np.asarray(q, float)
    profile = np.asarray(profile, float)
    background = np.asarray(background, float)
    valid = (q > 0) & np.isfinite(profile) & np.isfinite(background) & (background > 0)
    if not valid.any():
        return None
    order = np.argsort(q[valid])
    qv = q[valid][order]
    r = profile[valid][order] / background[valid][order]

    above = r >= ratio
    cross = np.where(above[:-1] & ~above[1:])[0]   # >=ratio then <ratio
    if cross.size == 0:
        return None
    i = cross[0]
    q0, q1, r0, r1 = qv[i], qv[i + 1], r[i], r[i + 1]
    return q0 if r1 == r0 else q0 + (ratio - r0) * (q1 - q0) / (r1 - r0)


def load_sample_groups():
    """Map sample name -> sorted [(run, path)] over scratch/saved_hits.

    Reuses merge_cxi's grouping (the one place that defines the sample-name
    convention). Imported lazily so --help and explicit --cxi/run-number use work
    without EXP_PREFIX set or merge_cxi's dependencies importable."""
    import merge_cxi
    return merge_cxi.group_by_sample(merge_cxi.SAVED_HITS_DIR), merge_cxi.SAVED_HITS_DIR


def parse_args():
    p = argparse.ArgumentParser(
        description='Show the good-hit powder radial profile integrated over '
                    'several runs (as in DAMNIT, but summed across runs).')
    p.add_argument('runs', type=int, nargs='*',
                   help='run numbers -> r{run:04d}_hits.cxi in scratch/saved_hits')
    p.add_argument('-s', '--sample', action='append', default=[],
                   help='select every hit CXI with this sample name (repeatable)')
    p.add_argument('-c', '--cxi', action='append', default=[],
                   help='explicit CXI file path (repeatable; adds to the selection)')
    p.add_argument('--list-samples', action='store_true',
                   help='list available sample names and their runs, then exit')
    p.add_argument('-m', '--model',
                   help='3D model h5 to overlay (default scratch/models/Ery_075nm.h5; '
                        'pass "none" to skip)')
    p.add_argument('--sb-ratio', type=float, default=0.6,
                   help='mark the resolution where signal/background falls through '
                        'this value (default 0.6; needs the bg subtraction)')
    p.add_argument('--png', help='save the figure to this path instead of showing it')
    args = p.parse_args()

    if args.list_samples:
        groups, directory = load_sample_groups()
        if not groups:
            print(f'no hit CXI files with a sample name in {directory}')
        for name in sorted(groups):
            runs = [r for r, _ in groups[name]]
            print(f'{name}: {len(runs)} runs {runs}')
        raise SystemExit(0)

    args.cxi_files = []
    labels = []
    if args.runs:
        from constants import PREFIX
        for run in args.runs:
            args.cxi_files.append(f'{PREFIX}scratch/saved_hits/r{run:04d}_hits.cxi')
        labels.append('+'.join(str(r) for r in args.runs))
    if args.sample:
        groups, directory = load_sample_groups()
        for sample in args.sample:
            if sample not in groups:
                avail = ', '.join(sorted(groups)) or '(none)'
                p.error(f'sample {sample!r} not found in {directory}; '
                        f'available: {avail}')
            runs = [r for r, _ in groups[sample]]
            args.cxi_files.extend(path for _, path in groups[sample])
            labels.append(f'{sample} (runs {"+".join(str(r) for r in runs)})')
    args.cxi_files.extend(args.cxi)
    if args.cxi:
        labels.append(f'{len(args.cxi)} files')
    if not args.cxi_files:
        p.error('give at least one run number, --sample, or --cxi')

    # a run may be named more than once (e.g. by number and by sample); de-dupe
    # so its powder is summed once, preserving selection order.
    seen = set()
    args.cxi_files = [pth for pth in args.cxi_files
                      if not (pth in seen or seen.add(pth))]
    args.label = '; '.join(labels)

    if args.model is None:
        from constants import PREFIX
        args.model = f'{PREFIX}scratch/models/Ery_075nm.h5'
    elif args.model.lower() == 'none':
        args.model = None
    return args


if __name__ == '__main__':
    args = parse_args()

    q_p, P, raw, background, n_used = integrate_powder_profile(args.cxi_files)
    n_skipped = len(args.cxi_files) - n_used
    print(f'integrated {n_used}/{len(args.cxi_files)} run(s)'
          f'{f" ({n_skipped} skipped)" if n_skipped else ""}: {P.shape[0]} q bins, '
          f'q_max = {q_p.max():.4g} 1/m (d_min = {1/q_p.max()*1e9:.3g} nm); '
          f'background {"subtracted" if background is not None else "unavailable"}')

    q_m = M = None
    if args.model is not None:
        q_m, M = prp.compute_model_profile(args.model, qmax=float(q_p.max()))

    fig = prp.make_powder_profile_figure(q_p, P, raw, background, q_m, M, args.label)

    # mark the resolution where signal/background = sb_ratio
    qc = sb_crossing_q(q_p, P, background, args.sb_ratio)
    if qc is not None:
        d_nm = 1e9 / qc
        ax = fig.axes[0]   # main axes (created first by make_powder_profile_figure)
        ax.axvline(qc * 1e-9, color='red', lw=1, ls=':',
                   label=f'S/B = {args.sb_ratio:g}  (d = {d_nm:.2g} nm)')
        ax.legend()
        print(f'signal/background = {args.sb_ratio:g} at q = {qc:.4g} 1/m '
              f'(resolution d = {d_nm:.3g} nm)')
    elif background is None:
        print('no background subtraction available; cannot mark signal/background ratio')
    else:
        print(f'signal/background never falls through {args.sb_ratio:g} '
              'within the q-range; no resolution line drawn')

    if args.png:
        fig.savefig(args.png)
        print(f'saved figure to {args.png}')
    else:
        import matplotlib.pyplot as plt
        plt.show()
