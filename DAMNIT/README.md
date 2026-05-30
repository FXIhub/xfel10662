# DAMNIT context for xfel10662

`context.py` is the [DAMNIT](https://damnit.readthedocs.io/) context file
for the SPB/p010662 single-particle imaging experiment. It defines the
per-run variables (start/end timestamps, train counts, eventually hit
rates, sample, pipeline status, ...) that DAMNIT computes and displays.

It is developed here under version control. To exercise it on Maxwell,
either run a **rehearsal** against old proposal data (Section A), or
**deploy** it for the live beamtime (Section B).

---

## Prerequisites

- SSH alias `max-exfl-display` to Maxwell (already in `~/.ssh/config`).
- On Maxwell, load DAMNIT with:
  ```sh
  source /etc/profile.d/modules.sh
  module load exfel damnit/stable
  ```
- The DAMNIT module name on Maxwell is `damnit/stable`. The `exfel-python`
  module is currently broken (missing `termcolor`); don't use it.
- The CLI is `damnit` (the older `amore-proto` still works but prints a
  deprecation warning).

---

## A. Rehearsal: test the context on old data (proposal 7927)

The rehearsal exercises the full code path — DAMNIT db, context-file
variables, eventually cluster jobs — against an old proposal that still
has data on disk. **The web frontend at https://damnit.xfel.eu cannot
see a sandbox database** (it is keyed to the canonical
`<proposal>/usr/Shared/amore/` path). So during rehearsal you view
results via the Python API or the PyQt GUI.

### A.1 (Laptop) Sync this repo to Maxwell

Run from the repo root:

```sh
REPO_ON_MAXWELL=/gpfs/exfel/exp/SPB/202405/p007927/scratch/amorgan/xfel10662
rsync -av --exclude=__pycache__ --exclude='*.pyc' \
    /home/andyofmelbourne/Documents/git_repos/xfel10662/ \
    max-exfl-display:$REPO_ON_MAXWELL/
```

### A.2 (Maxwell) Initialise a sandbox DAMNIT database

```sh
ssh max-exfl-display
source /etc/profile.d/modules.sh
module load exfel damnit/stable

SANDBOX=/gpfs/exfel/exp/SPB/202405/p007927/scratch/amorgan/damnit-sandbox
mkdir -p $SANDBOX
damnit init $SANDBOX --proposal 7927
```

### A.3 (Maxwell) Symlink the sandbox context to the repo copy

So edits in the repo flow into the sandbox automatically:

```sh
REPO_ON_MAXWELL=/gpfs/exfel/exp/SPB/202405/p007927/scratch/amorgan/xfel10662
rm $SANDBOX/context.py
ln -s $REPO_ON_MAXWELL/DAMNIT/context.py $SANDBOX/context.py
```

### A.4 (Maxwell) Reprocess one or more runs

```sh
cd $SANDBOX
# --direct runs in subprocesses on this node, no Slurm
# --watch shows live output
damnit reprocess --watch --direct 100
# or several:
damnit reprocess --watch --direct 100 101 102 103
# or every existing run (slow):
damnit reprocess --watch --direct all
```

### A.5 View results

**Python API** — quickest for a smoke check:

```sh
python -c "
from damnit import Damnit
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
df = Damnit('$SANDBOX').table(with_titles=True)
print(df)
"
```

**PyQt GUI** — the real interactive table (table, plots, comments).
Requires X11 forwarding:

```sh
# from laptop
ssh -X max-exfl-display
source /etc/profile.d/modules.sh
module load exfel damnit/stable
cd /gpfs/exfel/exp/SPB/202405/p007927/scratch/amorgan/damnit-sandbox
damnit gui --no-kafka .
```

### A.6 Iterate

After editing `context.py` locally:

```sh
# laptop
rsync -av --exclude=__pycache__ --exclude='*.pyc' \
    /home/andyofmelbourne/Documents/git_repos/xfel10662/ \
    max-exfl-display:/gpfs/exfel/exp/SPB/202405/p007927/scratch/amorgan/xfel10662/

# maxwell
ssh max-exfl-display
source /etc/profile.d/modules.sh && module load exfel damnit/stable
cd /gpfs/exfel/exp/SPB/202405/p007927/scratch/amorgan/damnit-sandbox
damnit reprocess --watch --direct <runs>      # re-run with new context
# or if only the context CHANGED (not new runs):
damnit read-context                            # re-reads context, refreshes view
```

If you add or rename variables and the sqlite view doesn't update,
the simplest reset is to wipe and re-init:

```sh
rm -rf $SANDBOX/*
damnit init $SANDBOX --proposal 7927
rm $SANDBOX/context.py
ln -s $REPO_ON_MAXWELL/DAMNIT/context.py $SANDBOX/context.py
damnit reprocess --watch --direct <runs>
```

---

## B. Deploy for the new experiment (proposal 10662)

Run this **when the beamtime is allocated** and you want DAMNIT to start
auto-processing runs as they arrive. Once the canonical
`usr/Shared/amore/` is in place, it appears at
https://damnit.xfel.eu/app/proposal/10662 automatically.

### B.1 (Laptop) Sync the repo to your personal area on the proposal

```sh
DEST=/gpfs/exfel/exp/SPB/202601/p010662/usr/Shared/amorgan/xfel10662
rsync -av --exclude=__pycache__ --exclude='*.pyc' \
    /home/andyofmelbourne/Documents/git_repos/xfel10662/ \
    max-exfl-display:$DEST/
```

### B.2 (Maxwell) Initialise the canonical DAMNIT directory

```sh
ssh max-exfl-display
source /etc/profile.d/modules.sh
module load exfel damnit/stable
damnit init 10662
# creates /gpfs/exfel/exp/SPB/202601/p010662/usr/Shared/amore/
```

### B.3 (Maxwell) Symlink the context file

```sh
CANON=/gpfs/exfel/exp/SPB/202601/p010662/usr/Shared/amore
REPO=/gpfs/exfel/exp/SPB/202601/p010662/usr/Shared/amorgan/xfel10662
rm $CANON/context.py
ln -s $REPO/DAMNIT/context.py $CANON/context.py
```

### B.4 (Maxwell) Configure beamtime-specific settings

```sh
cd $CANON
damnit db-config slurm_partition upex
# during the live beamtime, add the reservation:
damnit db-config slurm_reservation upex_010662
```

### B.5 Browser access

Navigate to https://damnit.xfel.eu/app/proposal/10662 — the central
listener (`xdamnprd`) monitors all proposals and will start processing
runs as they migrate. If for any reason auto-detection lags, you can
register the db explicitly:

```sh
damnit listener add 10662 $CANON
```

### B.6 Manual reprocess

If a context-file change needs applying to runs that have already
arrived:

```sh
cd $CANON
damnit reprocess --watch <runs>     # uses Slurm (no --direct)
```

---

## Gotchas (what we learned during prep)

- **Reserved variable name**: `start_time` is a built-in DAMNIT column on
  `run_info` (Unix-seconds, populated from migration metadata, displayed as
  `Timestamp`). Defining a user `@Variable` named `start_time` overwrites
  that slot and breaks `Damnit().table()`. Use distinct names like
  `t_first_train`. Other run_info names worth avoiding: `added_at`,
  `comment`.
- **Renaming variables**: the sqlite `runs` view is baked from the context
  at first read. After renaming, either run `damnit read-context` or wipe
  and re-init the db.
- **Browser cannot see sandbox dbs**: the web frontend resolves proposal
  number → `<proposal>/usr/Shared/amore/` only. Sandboxes elsewhere are
  invisible. Use the Python API or PyQt GUI for rehearsal viewing.
- **Cross-proposal reprocess tags rows with the source proposal**:
  `damnit reprocess --proposal 7927 100` stores `run_info.proposal = 7927`
  regardless of the host db's `metameta.proposal`. We tried mirroring p7927
  data into a p010662-canonical db for a browser preview and the rows
  didn't render — punted for now.
- **`exfel-python` module is broken**: missing `termcolor`. Use
  `damnit/stable`.

---

## Files

- `context.py` — the DAMNIT context file (variable definitions).
- `README.md` — this file.
