#!/bin/bash

# call: ./submit_cxi.sh <run_no>
# eg:   ./submit_events.sh 2

source /etc/profile.d/modules.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PARENT_DIR=$(dirname $SCRIPT_DIR)
source $PARENT_DIR/source_at_maxwell.sh

cd $SCRIPT_DIR

RES_ARG=""
[ -n "${RESERVATION}" ] && RES_ARG="--reservation=${RESERVATION}"
sbatch ${RES_ARG} <<EOT
#!/bin/bash

#SBATCH --array=${1}
#SBATCH --time=01:00:00
# --export=NONE: when launched from a nested DAMNIT slurm job, --export=ALL
# leaks DAMNIT's pixi env (PATH/PYTHONPATH/MODULEPATH) into this job and
# silently breaks \`module load exfel-python\` (the exfel modulefile isn't on
# DAMNIT's MODULEPATH; python keeps pointing at DAMNIT's pixi, which lacks
# extra_geom/tqdm). Start clean; source_at_maxwell.sh sets the env we need.
#SBATCH --export=NONE
#SBATCH -J cxi-${EXP_ID}
#SBATCH -o ${EXP_PREFIX}/scratch/log/cxi-${EXP_ID}-%A-%a.out
#SBATCH -e ${EXP_PREFIX}/scratch/log/cxi-${EXP_ID}-%A-%a.out
#SBATCH --partition=${PARTITION}

# exit on first error
set -e

# print commands and run them
set -o xtrace

source /etc/profile.d/modules.sh
source $PARENT_DIR/source_at_maxwell.sh

run=\${SLURM_ARRAY_TASK_ID}
echo ${1} run = \${run}

# Run the CXI writer and the background scan in parallel. The background
# script writes only to a sidecar (r{run}_hits.cxi.bg.h5) in this phase, so
# neither side touches the CXI file concurrently. A small --merge step
# afterwards folds the sidecar into the CXI file.
python make_cxi_file.py --nproc=16 \${run} &
MAKE_PID=\$!
python add_background_cxi.py --nproc=16 \${run} &
BG_PID=\$!

wait \$MAKE_PID
wait \$BG_PID

python add_background_cxi.py \${run} --merge

echo cxi done

EOT
