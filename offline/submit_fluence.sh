#!/bin/bash

# call: ./submit_fluence.sh <run_no>            # one run
#       ./submit_fluence.sh 12,13,17            # a comma list -> one array, one task per run
#
# Re-fits a background-aware per-frame fluence weight w_d for Ery sizing runs and
# writes the per-event incident fluence (photons/m^2) back into r####_hits.cxi.
#
# Runs *in the existing sizing working dir* scratch/sizing/r####/ (set up by
# submit_sizing.sh): it reuses that run's config.pickle, model, mask, run.cxi and
# logR matrix, so run sizing first. Only the per-frame w_d is refit.
#
# To run on a different proposal's already-sized runs (e.g. the previous beamtime
# p010996), point PROP_PREFIX at its proposal path -- everything (working dir,
# logs, the ../../models pickle symlink) then resolves under that proposal:
#       PROP_PREFIX=/gpfs/exfel/exp/SPB/202604/p010996/ ./submit_fluence.sh 26,27,28

source /etc/profile.d/modules.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PARENT_DIR=$(dirname $SCRIPT_DIR)
source $PARENT_DIR/source_at_maxwell.sh

# proposal whose scratch/sizing + scratch/log to use (default: this proposal)
PROP_PREFIX="${PROP_PREFIX:-$EXP_PREFIX}"

cd $SCRIPT_DIR

# pin the cluster: a bare sbatch can otherwise resolve to the solaris cluster and
# reject the maxwell partition ("invalid partition specified") -- see notes.
export SLURM_CLUSTERS=maxwell

RES_ARG=""
[ -n "${RESERVATION}" ] && RES_ARG="--reservation=${RESERVATION}"
sbatch --wait ${RES_ARG} <<EOT
#!/bin/bash

#SBATCH --array=${1}
#SBATCH --time=04:00:00
##SBATCH --export=NONE
#SBATCH -J fluence-%a
#SBATCH -o ${PROP_PREFIX}/scratch/log/fluence-%a.out
#SBATCH -e ${PROP_PREFIX}/scratch/log/fluence-%a.out
#SBATCH --partition=${PARTITION}

# exit on first error
set -e

# print commands and run them
set -o xtrace

source /etc/profile.d/modules.sh
# Source for env vars (EXP_PREFIX, EXP_ID, PARTITION). The modules it loads
# are intentionally wiped by \`module purge\` below; we replace them with the
# EMC-specific set.
source $PARENT_DIR/source_at_maxwell.sh

run=\${SLURM_ARRAY_TASK_ID}
echo ${1} run = \${run}

# reuse the sizing working dir (do not create a fresh one)
zrun=r\$(printf "%04d" \${run}) # zero padded run number
WD=${PROP_PREFIX}/scratch/sizing/\${zrun}
if [ ! -d "\$WD" ]; then
    echo "ERROR: \$WD does not exist -- run sizing first"
    exit 1
fi

cp run_emc_fluence.py \${WD}/run_emc_fluence.py

cd \$WD

# the |F|^2 pickle the model was built from (for I_max recovery). The other
# inputs (class_model_0.h5, mask.h5, run.cxi, config.pickle, class_logR_0.h5) are
# already in place from the sizing run.
ln -sf ../../models/Ery_075_aligned_dq_2707310.0_N_192.pickle model_intensity.pickle


unset LD_PRELOAD
module purge
module load exfel exfel-python/202501
module load mpi/mvapich2-x86_64
# --export=NONE means ~/.bashrc isn't sourced, so conda isn't initialized.
# The legacy /software/anaconda3/5.2 path in ~/.bashrc no longer exists on
# compute nodes; conda now lives under /software/mamba/<version>/. Bump the
# version below when Maxwell rotates it.
source /software/mamba/2026.05/etc/profile.d/conda.sh
conda activate EMC
export PYTHONPATH=\$PYTHONPATH:/home/amorgan/EMC2/
export LD_LIBRARY_PATH=/usr/lib64/mvapich2/lib:\$LD_LIBRARY_PATH
export SLURM_CPU_BIND=verbose
export OMP_NUM_THREADS=\$SLURM_CPUS_ON_NODE
export MPICH_CPUMASK_DISPLAY=1
export MPICH_CPU_BIND_TYPE=mask
export MPICH_CPU_BIND_STEP=\$SLURM_CPUS_ON_NODE

which mpiexec
mpiexec --version

# srun (the mvapich2 launcher) aborts with "cpus-per-task set by two different
# environment variables SLURM_CPUS_PER_TASK != SLURM_TRES_PER_TASK" when both
# are present and disagree (Slurm >=23.11). Clear them; OMP_NUM_THREADS above
# already sets threading.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

mpirun -n \${SLURM_NTASKS:-1} python run_emc_fluence.py

echo fluence done

EOT
