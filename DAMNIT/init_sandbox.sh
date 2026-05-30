#!/bin/bash
source /etc/profile.d/modules.sh
module load exfel damnit/stable

mkdir -p $SANDBOX
damnit init $SANDBOX --proposal 7927

rm $SANDBOX/context.py
ln -s $REPO_ON_MAXWELL/DAMNIT/context.py $SANDBOX/context.py
