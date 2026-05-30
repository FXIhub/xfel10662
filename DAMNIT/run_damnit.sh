#!/bin/bash

source /etc/profile.d/modules.sh
module load exfel damnit/stable

cd $SANDBOX

# ./run_damnit.sh run_no
# --direct runs in subprocesses on this node, no Slurm
# --watch shows live output
# damnit reprocess --watch --direct 100
# or several:
# damnit reprocess --watch --direct 100 101 102 103
# or every existing run (slow):
# damnit reprocess --watch --direct all

damnit reprocess --in $SANDBOX --watch --direct $1

cd $REPO_ON_MAXWELL/DAMNIT
python print_damnit_db.py
