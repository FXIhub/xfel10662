export SANDBOX=/gpfs/exfel/exp/SPB/202405/p007927/scratch/amorgan/damnit-sandbox
export REPO_ON_MAXWELL=/gpfs/exfel/exp/SPB/202601/p010662/usr/Shared/xfel10662
export EXP_ID=007927
export EXP_PREFIX=/gpfs/exfel/exp/SPB/202405/p007927/

# for beamtime
#export PARTITION=upex-beamtime
#export RESERVATION=upex_${EXP_ID}

# for 6 months after beamtime
#export PARTITION=upex
# for more than 6 months after beamtime
export PARTITION=cfel
#export PARTITION=allgpu
export RESERVATION=

# If invoked from a slurm job submitted by DAMNIT (which runs under its own
# pixi env), --export=ALL leaks PYTHONPATH/CONDA_PREFIX/... from that env into
# the job, which makes the exfel-python module's python import the wrong
# site-packages (missing extra_geom/tqdm). Strip those before loading the
# module. PATH is left alone — the module load prepends its own bin dir so
# `python` resolves to the exfel-python interpreter.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV PIXI_PROJECT_ROOT PIXI_ENVIRONMENT_NAME

source /etc/profile.d/modules.sh
# Ensure /etc/modulefiles is on MODULEPATH: in slurm jobs autoinit may not add
# it (login shells get it via a different code path), and that's where the
# `exfel` modulefile lives. Without this, `module load exfel` silently
# no-ops and python imports fail.
module use /etc/modulefiles
module load exfel exfel-python
