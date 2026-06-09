"""Show the pipeline status (vds / cxi / classification / sizing) for a range of
runs as a table.

Status of each (run, process) is read from the slurm log
$EXP_PREFIX/scratch/log/{process}-{run}.out:

    nothing   log file does not exist
    done      log contains "{process} done" (written at the end of the job)
    error     log exists but has no "done" marker (and no job is running)
    running   a slurm array job for this (process, run) is queued or running

Usage:
    python show_pipeline_status.py 1 50        # runs 1..50 inclusive
    python show_pipeline_status.py 7           # single run
    python show_pipeline_status.py 1 50 --no-color
"""

import os
import argparse
import subprocess
from pathlib import Path

PROCESSES = ['vds', 'cxi', 'classification', 'sizing']

PREFIX = Path(os.environ["EXP_PREFIX"])
LOG_DIR = PREFIX / "scratch" / "log"

# ANSI colours keyed by status
COLOURS = {
    'nothing': '90',   # grey
    'done':    '32',   # green
    'error':   '31',   # red
    'running': '33',   # yellow
}


def log_status(process, run):
    """nothing / done / error from the log file alone."""
    log = LOG_DIR / f"{process}-{run}.out"
    if not log.is_file():
        return 'nothing'
    marker = f'{process} done'.lower()
    try:
        with open(log, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if marker in line.lower():
                    return 'done'
    except OSError:
        return 'error'
    return 'error'


def running_jobs():
    """Set of (process, run) currently queued or running in slurm.

    squeue -r expands array elements, so the job id looks like <jobid>_<taskid>
    where taskid is the run number, and the job name starts with the process
    name (e.g. cxi-7 / cxi-%a). Returns an empty set if squeue is unavailable.
    """
    active = set()
    try:
        out = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-r',
             '-o', '%i|%j'],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return active

    for line in out.splitlines():
        if '|' not in line:
            continue
        jobid, name = line.split('|', 1)
        if '_' not in jobid:
            continue
        task = jobid.rsplit('_', 1)[1]
        if not task.isdigit():          # skip array ranges like 12345_[7-9]
            continue
        process = name.split('-', 1)[0]
        if process in PROCESSES:
            active.add((process, int(task)))
    return active


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('start', type=int, help='first run number')
    parser.add_argument('end', type=int, nargs='?',
                        help='last run number (inclusive); default = start')
    parser.add_argument('--no-color', action='store_true',
                        help='disable ANSI colour')
    args = parser.parse_args()

    end = args.end if args.end is not None else args.start
    runs = range(args.start, end + 1)

    active = running_jobs()
    use_colour = not args.no_color and os.isatty(1)

    # build the table
    run_w = max(len('run'), len(str(end)))
    col_w = max(9, *(len(p) for p in PROCESSES))

    def cell(status):
        if use_colour:
            return f'\033[{COLOURS[status]}m{status:<{col_w}}\033[0m'
        return f'{status:<{col_w}}'

    header = f'{"run":>{run_w}}  ' + '  '.join(f'{p:<{col_w}}' for p in PROCESSES)
    print(header)
    print('-' * len(header))

    for run in runs:
        cells = []
        for p in PROCESSES:
            status = log_status(p, run)
            if status != 'done' and (p, run) in active:
                status = 'running'
            cells.append(cell(status))
        print(f'{run:>{run_w}}  ' + '  '.join(cells))


if __name__ == '__main__':
    main()
