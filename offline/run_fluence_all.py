"""
Submit the per-event fluence re-fit (submit_fluence.sh -> run_emc_fluence.py) for
every Erythrocruorin run.

Ery runs are discovered the same way merge_cxi does: every
scratch/saved_hits/r####_hits.cxi whose entry_1/sample_1/name matches the sample
(default 'Erythrocruorin'), which naturally excludes the beta runs. The run
numbers are handed to submit_fluence.sh as a single comma-separated sbatch array
spec (one array task per run), matching submit_sizing_beta.sh's --array=${1} idiom.

Usage
-----
    python run_fluence_all.py                      # submit all Erythrocruorin runs
    python run_fluence_all.py --dry-run            # just list the runs + command
    python run_fluence_all.py 12 13 17             # only these runs
    python run_fluence_all.py --sample "Erythrocruorin"
"""
import argparse
import subprocess
from pathlib import Path

import merge_cxi

SUBMIT = Path(__file__).resolve().parent / 'submit_fluence.sh'


def ery_runs(sample):
    groups = merge_cxi.group_by_sample(merge_cxi.SAVED_HITS_DIR)
    if sample not in groups:
        avail = ', '.join(sorted(groups)) or '(none)'
        raise SystemExit(f'sample {sample!r} not found in {merge_cxi.SAVED_HITS_DIR}; '
                         f'available: {avail}')
    return [run for run, _ in groups[sample]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', type=int, nargs='*',
                   help='only submit these run numbers (default: all sample runs)')
    p.add_argument('--sample', default='Erythrocruorin',
                   help="sample name to select (default 'Erythrocruorin')")
    p.add_argument('--dry-run', action='store_true',
                   help='print the runs and the submit command without submitting')
    args = p.parse_args()

    runs = sorted(set(args.runs)) if args.runs else ery_runs(args.sample)
    if not runs:
        raise SystemExit('no runs to submit')

    array = ','.join(str(r) for r in runs)
    print(f'{len(runs)} run(s): {runs}')
    print(f'{SUBMIT} {array}')

    if args.dry_run:
        return
    subprocess.run([str(SUBMIT), array], check=True)


if __name__ == '__main__':
    main()
