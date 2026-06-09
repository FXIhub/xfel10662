#!/bin/bash
#SBATCH --time=096:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
##SBATCH --constraint="A100"
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_010662
#SBATCH -J 2D_EMC
#SBATCH -o 2D_EMC.log
#SBATCH -e 2D_EMC.log
set -e
source /etc/profile.d/modules.sh
unset LD_PRELOAD
module purge
module load exfel exfel-python/202501
module load mpi/mvapich2-x86_64
conda activate EMC
#export SLURM_MPI_TYPE=pmi2
#export PYOPENCL_COMPILER_OUTPUT=1
export PYTHONPATH=$PYTHONPATH:/home/amorgan/EMC2/
export LD_LIBRARY_PATH=/usr/lib64/mvapich2/lib:$LD_LIBRARY_PATH

# Optional: Set SLURM to report what it is doing with binding
export SLURM_CPU_BIND=verbose

# -------------------------------------------------------------
# 1. Configure OpenCL/Threading Environment (Crucial Step)
# -------------------------------------------------------------
# Find out how many CPUs the system allocated to this task on each node.
# The variable $SLURM_CPUS_ON_NODE will hold the total number of cores/threads per node.

# Set OMP_NUM_THREADS (even if OpenCL uses its own threading, this is good practice)
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Optional: Set a vendor-specific OpenCL variable if needed
# export DPCPP_CPU_NUM_CUS=$SLURM_CPUS_ON_NODE

# -------------------------------------------------------------
# 2. Configure MPI/MPICH Binding
# -------------------------------------------------------------
# Use MPICH environment variables to ensure binding awareness
export MPICH_CPUMASK_DISPLAY=1
export MPICH_CPU_BIND_TYPE=mask
export MPICH_CPU_BIND_STEP=$SLURM_CPUS_ON_NODE
# NOTE: The above MPICH variables might vary slightly based on your MPICH version/build.

which mpiexec
mpiexec --version


# to prevent multiple ranks accessing the same file
#export PYOPENCL_NO_CACHE=1
#export PYTOOLS_PERSISTENT_DICT_SAFE_SYNC=False

# -------------------------------------------------------------
# 3. Launch the Job
# -------------------------------------------------------------
# Use mpiexec to launch the processes across the nodes.
# It reads the SLURM allocation to launch correctly.
mpirun -n $SLURM_NTASKS python run_emc_2D.py
echo finished

