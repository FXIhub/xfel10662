#!/bin/bash

# call: ./submit_sizing.sh <run_no>

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
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH -J sizing-${EXP_ID}
#SBATCH -o ${EXP_PREFIX}/scratch/log/sizing-${EXP_ID}-%A-%a.out
#SBATCH -e ${EXP_PREFIX}/scratch/log/sizing-${EXP_ID}-%A-%a.out
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

# make working directory
zrun=r\$(printf "%04d" \${run}) # zero padded run number
WD=${EXP_PREFIX}/scratch/sizing/\${zrun}
mkdir -p \$WD

cp config_sizing.py \${WD}/config.py
cp run_emc_sizing.py \${WD}/run_emc.py

cd \$WD

# symlink files
ln -sf ../../models/Ery_075nm.h5 class_model_0.h5

ln -sf ../../det/mask.h5 mask.h5

ln -sf ../../saved_hits/\${zrun}_hits.cxi run.cxi


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

mpirun -n \${SLURM_NTASKS:-1} python run_emc.py

echo sizing done

EOT

